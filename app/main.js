// Main features:
// - Precise absolute timestamps using sample indices (16kHz) relative to capture start
// - Word-level timestamps (Whisper) mapped to absolute capture timeline
// - Optional speaker diarization:
//   * simple online VAD -> speaker segment candidates
//   * local speaker embeddings from worker (feature-extraction)
//   * online clustering by cosine similarity threshold
//   * word-to-speaker assignment by overlap in time
//
// Notes:
// - This is best-effort diarization. Expect errors on noisy audio, overlapping speech, very short turns.
// - For reliable diarization you typically need a dedicated diarization model/pipeline; this is the best practical
//   “static web app” approach without server-side processing.

const $ = (id) => document.getElementById(id);

// UI
const btnStart = $("btnStart");
const btnStop = $("btnStop");
const btnClear = $("btnClear");
const btnTranscribe = $("btnTranscribe");

const btnExport = $("btnExport");
const btnExportTxt = $("btnExportTxt");

const asrModelSel = $("asrModel");
const devicePrefSel = $("devicePref");
const winSecIn = $("winSec");
const ovlSecIn = $("ovlSec");
const timestampsSel = $("timestamps");
const autoRunSel = $("autoRun");
const diarizeSel = $("diarize");
const spkThreshIn = $("spkThresh");

const transcriptTa = $("transcript");
const summaryTa = $("summary");
const diagOut = $("diagOut");

const autoScrollChk = $("autoScroll");
const showRawChk = $("showRaw");

const gpuPill = $("gpuPill");
const asrPill = $("asrPill");
const diaPill = $("diaPill");
const owuiPill = $("owuiPill");

// OpenWebUI
const owBaseUrlIn = $("owBaseUrl");
const owApiKeyIn = $("owApiKey");
const owModelSel = $("owModel");
const sumEverySecIn = $("sumEverySec");
const sumStyleSel = $("sumStyle");
const owStreamSel = $("owStream");
const btnLoadModels = $("btnLoadModels");
const btnTestOWUI = $("btnTestOWUI");
const btnSummarize = $("btnSummarize");
const btnClearSummary = $("btnClearSummary");

// Constants
const TARGET_SR = 16000;

// Capture/Audio
let displayStream = null;
let audioCtx = null;
let workletNode = null;

// Absolute time anchor
let captureStartEpochMs = null;     // wallclock when capture started
let firstSampleStart = null;        // first sample index received from worklet (usually 0)

// Ring buffer with absolute sample indices
// We store audio as a list of blocks to avoid O(n) copies on each append.
let pcmBlocks = []; // each { startSample:number, pcm:Float32Array, rms:number }
let totalBufferedSamples = 0;

// ASR job handling
let worker = null;
let asrReady = false;
let spkReady = false;
let asrDevice = null;
let spkDevice = null;

// ASR inflight window (single inflight to avoid buffer loss)
let inflightAsr = null; // { id, winStartSample, winEndSample, sentAtMs }

// Transcript structure
// We store word-level tokens when available, else store chunk as a segment.
let transcriptWords = []; // { t0, t1, text, speakerId|null }
let transcriptSegments = []; // { t0, t1, text, rawText, words?:[], speakerId?: }

// Diarization
// VAD state on downsampled chunks
const VAD = {
  enabled: false,
  // parameters (tunable)
  rmsThreshold: 0.015,  // naive RMS threshold (depends on system audio; may need UI later)
  hangoverMs: 400,      // keep speech on briefly after falling below threshold
  minSegMs: 900,        // ignore very short segments
  maxSegMs: 12000,      // cap segment size for embeddings
  // state
  speechOn: false,
  lastSpeechMs: 0,
  currentSeg: null,     // { t0, t1, blocks:[...], startSample, endSample }
  segments: [],         // finalized candidate segments for speaker embedding {t0,t1,startSample,endSample, pcm}
};

// Speaker clusters
let speakerClusters = []; // { id, centroid:Float32Array, count:number }
let nextSpeakerId = 1;

// Summary
let lastSummarizedChar = 0;
let summaryTimer = null;

// Initialization
initOnce();

function initOnce() {
  const saved = safeParse(localStorage.getItem("local-asr-settings") || "{}");
  if (saved?.owBaseUrl) owBaseUrlIn.value = saved.owBaseUrl;
  if (saved?.owModel) setTimeout(() => setSelectValue(owModelSel, saved.owModel), 0);
  if (saved?.sumEverySec) sumEverySecIn.value = saved.sumEverySec;
  if (saved?.sumStyle) setSelectValue(sumStyleSel, saved.sumStyle);
  if (saved?.asrModel) setSelectValue(asrModelSel, saved.asrModel);
  if (saved?.devicePref) setSelectValue(devicePrefSel, saved.devicePref);
  if (saved?.winSec) winSecIn.value = saved.winSec;
  if (saved?.ovlSec) ovlSecIn.value = saved.ovlSec;
  if (saved?.timestamps) setSelectValue(timestampsSel, saved.timestamps);
  if (saved?.autoRun) setSelectValue(autoRunSel, saved.autoRun);
  if (saved?.owStream) setSelectValue(owStreamSel, saved.owStream);
  if (saved?.diarize) setSelectValue(diarizeSel, saved.diarize);
  if (saved?.spkThresh) spkThreshIn.value = saved.spkThresh;

  updateGpuPill();
  setDiaPillFromState();
  logDiag(`Loaded. Origin=${location.origin}`);

  btnStart.onclick = startCapture;
  btnStop.onclick = stopCapture;
  btnClear.onclick = clearAll;
  btnTranscribe.onclick = manualTranscribe;

  btnLoadModels.onclick = loadOpenWebUIModels;
  btnTestOWUI.onclick = testOpenWebUI;
  btnSummarize.onclick = () => summarizeNow(false);
  btnClearSummary.onclick = () => summaryTa.value = "";

  btnExport.onclick = exportJson;
  btnExportTxt.onclick = exportTxt;

  showRawChk.addEventListener("change", renderTranscript);

  [
    owBaseUrlIn, sumEverySecIn, sumStyleSel,
    asrModelSel, devicePrefSel, winSecIn, ovlSecIn, timestampsSel,
    autoRunSel, owStreamSel, diarizeSel, spkThreshIn
  ].forEach(el => el.addEventListener("change", persistSettings));

  persistSettings();
}

function persistSettings() {
  const obj = {
    owBaseUrl: owBaseUrlIn.value.trim(),
    owModel: owModelSel.value,
    sumEverySec: Number(sumEverySecIn.value) || 60,
    sumStyle: sumStyleSel.value,
    asrModel: asrModelSel.value,
    devicePref: devicePrefSel.value,
    winSec: Number(winSecIn.value) || 15,
    ovlSec: Number(ovlSecIn.value) || 3,
    timestamps: timestampsSel.value,
    autoRun: autoRunSel.value,
    owStream: owStreamSel.value,
    diarize: diarizeSel.value,
    spkThresh: Number(spkThreshIn.value) || 0.78,
  };
  localStorage.setItem("local-asr-settings", JSON.stringify(obj));
}

function updateGpuPill() {
  gpuPill.textContent = `GPU: ${navigator.gpu ? "WebGPU available" : "no WebGPU"}`;
}

function setAsrPill(text, kind = "neutral") {
  asrPill.textContent = `ASR: ${text}`;
  asrPill.style.color = pillColor(kind);
}

function setDiaPill(text, kind = "neutral") {
  diaPill.textContent = `Diarization: ${text}`;
  diaPill.style.color = pillColor(kind);
}

function setOwuiPill(text, kind = "neutral") {
  owuiPill.textContent = `OpenWebUI: ${text}`;
  owuiPill.style.color = pillColor(kind);
}

function pillColor(kind) {
  return kind === "good" ? "var(--good)" :
         kind === "warn" ? "var(--warn)" :
         kind === "bad"  ? "var(--bad)"  : "var(--muted)";
}

function setDiaPillFromState() {
  const on = diarizeSel.value === "on";
  setDiaPill(on ? "on" : "off", on ? "warn" : "neutral");
}

function logDiag(line) {
  const ts = new Date().toISOString();
  diagOut.textContent = `${ts}  ${line}\n` + diagOut.textContent;
}

function safeParse(s) { try { return JSON.parse(s); } catch { return null; } }
function setSelectValue(sel, v) {
  if (!v) return;
  for (const opt of sel.options) {
    if (opt.value === v) { sel.value = v; return; }
  }
}

function clearAll() {
  pcmBlocks = [];
  totalBufferedSamples = 0;
  inflightAsr = null;

  captureStartEpochMs = null;
  firstSampleStart = null;

  transcriptWords = [];
  transcriptSegments = [];

  VAD.speechOn = false;
  VAD.currentSeg = null;
  VAD.segments = [];

  speakerClusters = [];
  nextSpeakerId = 1;

  transcriptTa.value = "";
  summaryTa.value = "";
  lastSummarizedChar = 0;

  logDiag("Cleared state.");
  setDiaPillFromState();
}

// ----------- Capture start/stop ------------

async function startCapture() {
  try {
    btnStart.disabled = true;
    btnStop.disabled = false;
    btnClear.disabled = true;
    btnTranscribe.disabled = true;

    setAsrPill("initializing…", "warn");
    setDiaPillFromState();

    await ensureWorker();
    await initAsr();

    // Initialize diarization pipeline if enabled
    VAD.enabled = diarizeSel.value === "on";
    if (VAD.enabled) {
      await initSpeakerEmbedding();
      setDiaPill(`ready (${spkDevice})`, "good");
    } else {
      setDiaPill("off", "neutral");
    }

    displayStream = await navigator.mediaDevices.getDisplayMedia({
      video: true,
      audio: true,
    });

    const audioTrack = displayStream.getAudioTracks()[0];
    if (!audioTrack) throw new Error("No audio track captured. Ensure you share system audio.");

    captureStartEpochMs = Date.now();
    firstSampleStart = null;

    await startAudioPipeline(new MediaStream([audioTrack]));
    startSummaryTimer();

    btnClear.disabled = false;
    btnTranscribe.disabled = (autoRunSel.value === "on");
    setAsrPill(`ready (${asrDevice})`, "good");
    logDiag("Capture started.");
  } catch (e) {
    logDiag(`Start error: ${String(e)}`);
    setAsrPill("error", "bad");
    setDiaPill("error", "bad");
    btnStart.disabled = false;
    btnStop.disabled = true;
    btnClear.disabled = false;
    btnTranscribe.disabled = true;
  }
}

async function stopCapture() {
  try {
    btnStop.disabled = true;
    stopSummaryTimer();

    if (workletNode) {
      try { workletNode.port.onmessage = null; workletNode.disconnect(); } catch {}
      workletNode = null;
    }
    if (audioCtx) {
      try { await audioCtx.close(); } catch {}
      audioCtx = null;
    }

    if (displayStream) {
      for (const t of displayStream.getTracks()) t.stop();
      displayStream = null;
    }

    // Flush VAD segment if in progress
    if (VAD.enabled) finalizeVadSegmentIfAny(true);

    btnStart.disabled = false;
    btnStop.disabled = true;
    btnClear.disabled = false;
    btnTranscribe.disabled = false;

    setAsrPill("idle");
    setDiaPill(VAD.enabled ? "idle" : "off");
    logDiag("Capture stopped.");
  } catch (e) {
    logDiag(`Stop error: ${String(e)}`);
  }
}

// ----------- Worker init ------------

async function ensureWorker() {
  if (worker) return;

  worker = new Worker("./asr-worker.js", { type: "module" });

  worker.onmessage = (e) => {
    const msg = e.data || {};
    if (msg.type === "asr_status") {
      if (msg.status === "loading") setAsrPill("loading model…", "warn");
      else if (msg.status === "webgpu_failed_fallback_wasm") {
        logDiag(`ASR WebGPU failed → WASM. ${msg.error}`);
        setAsrPill("fallback wasm…", "warn");
      } else if (msg.status === "ready") setAsrPill(`ready (${asrDevice})`, "good");
    } else if (msg.type === "asr_ready") {
      asrReady = true;
      asrDevice = msg.device;
      setAsrPill(`ready (${asrDevice})`, "good");
      logDiag(`ASR ready: model=${msg.model} device=${msg.device}`);
    } else if (msg.type === "asr_result") {
      handleAsrResult(msg);
    } else if (msg.type === "asr_error") {
      logDiag(`ASR error: ${msg.error}`);
      setAsrPill("error", "bad");
      if (inflightAsr?.id === msg.id) inflightAsr = null;
    } else if (msg.type === "spk_status") {
      if (msg.status === "loading") setDiaPill("loading…", "warn");
      else if (msg.status === "webgpu_failed_fallback_wasm") {
        logDiag(`SPK WebGPU failed → WASM. ${msg.error}`);
        setDiaPill("fallback wasm…", "warn");
      } else if (msg.status === "ready") setDiaPill(`ready (${spkDevice})`, "good");
    } else if (msg.type === "spk_ready") {
      spkReady = true;
      spkDevice = msg.device;
      setDiaPill(`ready (${spkDevice})`, "good");
      logDiag(`SPK ready: model=${msg.model} device=${msg.device}`);
    } else if (msg.type === "spk_embed") {
      handleSpeakerEmbedding(msg);
    } else if (msg.type === "spk_error") {
      logDiag(`SPK error: ${msg.error}`);
      setDiaPill("error", "bad");
    }
  };
}

async function initAsr() {
  asrReady = false;
  const model = asrModelSel.value;
  const devicePref = devicePrefSel.value;
  worker.postMessage({ type: "init_asr", model, devicePref });
  await waitFor(() => asrReady, 120000, "ASR init timeout");
}

async function initSpeakerEmbedding() {
  spkReady = false;

  // You should validate/tune this model for your environment.
  // If you have a known-good speaker-verification model you prefer, swap it here.
  const spkModel = "Xenova/wavlm-base-plus"; // baseline feature extractor; not “true diarization” but workable for clustering
  const devicePref = devicePrefSel.value;

  worker.postMessage({ type: "init_spk", model: spkModel, devicePref });
  await waitFor(() => spkReady, 180000, "Speaker embedding init timeout");
}

function waitFor(pred, timeoutMs, msg) {
  const t0 = performance.now();
  return new Promise((resolve, reject) => {
    const tick = async () => {
      if (pred()) return resolve();
      if (performance.now() - t0 > timeoutMs) return reject(new Error(msg));
      setTimeout(tick, 50);
    };
    tick();
  });
}

// ----------- Audio pipeline ------------

async function startAudioPipeline(audioStream) {
  audioCtx = new AudioContext();
  await audioCtx.audioWorklet.addModule("./pcm-worklet.js");

  const src = audioCtx.createMediaStreamSource(audioStream);
  workletNode = new AudioWorkletNode(audioCtx, "pcm-capture");

  // Keep alive without feedback
  const gain = audioCtx.createGain();
  gain.gain.value = 0;

  src.connect(workletNode);
  workletNode.connect(gain);
  gain.connect(audioCtx.destination);

  workletNode.port.onmessage = (e) => {
    const { pcm, startSample, rms } = e.data || {};
    if (!pcm || pcm.length === 0) return;

    if (firstSampleStart === null) firstSampleStart = startSample;

    pcmBlocks.push({ startSample, pcm, rms });
    totalBufferedSamples += pcm.length;

    if (VAD.enabled) vadObserveChunk(startSample, pcm.length, rms);

    if (autoRunSel.value === "on") maybeTranscribe();
    else btnTranscribe.disabled = false;
  };
}

function getWindowParams() {
  const winSec = clamp(Number(winSecIn.value) || 15, 5, 60);
  const ovlSec = clamp(Number(ovlSecIn.value) || 3, 0, Math.min(10, winSec - 1));
  return {
    winSec, ovlSec,
    winSamples: Math.floor(winSec * TARGET_SR),
    ovlSamples: Math.floor(ovlSec * TARGET_SR)
  };
}

function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }

// Pull a contiguous window from pcmBlocks into a Float32Array, based on absolute sample indices.
// Returns { audio, startSample, endSample } or null.
function readWindowFromBlocks(winSamples) {
  if (pcmBlocks.length === 0) return null;

  const earliest = pcmBlocks[0].startSample;
  const latestBlock = pcmBlocks[pcmBlocks.length - 1];
  const latestEnd = latestBlock.startSample + latestBlock.pcm.length;

  if ((latestEnd - earliest) < winSamples) return null;

  const startSample = earliest;
  const endSample = startSample + winSamples;

  const out = new Float32Array(winSamples);

  // Copy samples from blocks that overlap [startSample, endSample)
  let outPos = 0;
  let i = 0;
  while (outPos < winSamples && i < pcmBlocks.length) {
    const b = pcmBlocks[i];
    const b0 = b.startSample;
    const b1 = b.startSample + b.pcm.length;

    // no overlap
    if (b1 <= startSample) { i++; continue; }
    if (b0 >= endSample) break;

    const copyStart = Math.max(b0, startSample);
    const copyEnd = Math.min(b1, endSample);

    const srcOffset = copyStart - b0;
    const dstOffset = copyStart - startSample;
    const n = copyEnd - copyStart;

    out.set(b.pcm.subarray(srcOffset, srcOffset + n), dstOffset);
    outPos += n;
    i++;
  }

  return { audio: out, startSample, endSample };
}

// Drop samples from pcmBlocks to keep only overlap
function consumeToKeepOverlap(keepStartSample) {
  // Remove blocks entirely before keepStartSample; trim the first overlapping block.
  while (pcmBlocks.length && (pcmBlocks[0].startSample + pcmBlocks[0].pcm.length) <= keepStartSample) {
    const b = pcmBlocks.shift();
    totalBufferedSamples -= b.pcm.length;
  }
  if (!pcmBlocks.length) return;

  const b0 = pcmBlocks[0];
  if (b0.startSample < keepStartSample) {
    const cut = keepStartSample - b0.startSample;
    b0.pcm = b0.pcm.subarray(cut);
    totalBufferedSamples -= cut;
    b0.startSample = keepStartSample;
  }
}

// ----------- Transcription loop ------------

function manualTranscribe() {
  btnTranscribe.disabled = true;
  maybeTranscribe();
}

function maybeTranscribe() {
  if (!displayStream) return;
  if (!asrReady) return;
  if (inflightAsr) return; // single inflight policy

  const { winSamples, ovlSamples } = getWindowParams();
  const win = readWindowFromBlocks(winSamples);
  if (!win) {
    btnTranscribe.disabled = true;
    return;
  }

  // Send to worker
  const id = Date.now() + Math.random();
  inflightAsr = { id, winStartSample: win.startSample, winEndSample: win.endSample, sentAtMs: Date.now() };

  worker.postMessage({
    type: "transcribe",
    id,
    audio: win.audio,
    samplingRate: TARGET_SR,
    timestamps: timestampsSel.value,
  }, [win.audio.buffer]);

  // Keep overlap region
  const keepStart = win.endSample - ovlSamples;
  consumeToKeepOverlap(keepStart);
}

// ----------- Timestamp helpers ------------

// Convert absolute sample index → absolute timestamp in milliseconds since epoch
function sampleToEpochMs(sampleIndex) {
  if (captureStartEpochMs === null || firstSampleStart === null) return null;
  const relSamples = sampleIndex - firstSampleStart;
  const relMs = (relSamples / TARGET_SR) * 1000;
  return captureStartEpochMs + relMs;
}

// Format epoch ms into [HH:MM:SS.mmm] relative to capture start
function formatRelTime(epochMs) {
  if (captureStartEpochMs === null) return "[--:--:--.---]";
  const rel = Math.max(0, epochMs - captureStartEpochMs);
  const h = Math.floor(rel / 3600000);
  const m = Math.floor((rel % 3600000) / 60000);
  const s = Math.floor((rel % 60000) / 1000);
  const ms = Math.floor(rel % 1000);
  return `[${pad2(h)}:${pad2(m)}:${pad2(s)}.${pad3(ms)}]`;
}
function pad2(n) { return String(n).padStart(2, "0"); }
function pad3(n) { return String(n).padStart(3, "0"); }

// ----------- Handle ASR results ------------

function handleAsrResult({ id, text, chunks, ms }) {
  if (!inflightAsr || inflightAsr.id !== id) {
    logDiag(`ASR result for unknown job id=${id} (ignored).`);
    return;
  }

  const winStart = inflightAsr.winStartSample;
  const winEnd = inflightAsr.winEndSample;
  inflightAsr = null;

  const t0Epoch = sampleToEpochMs(winStart);
  const t1Epoch = sampleToEpochMs(winEnd);

  if (!text) {
    logDiag(`ASR empty (${ms}ms).`);
    // Continue if buffered
    if (autoRunSel.value === "on") maybeTranscribe();
    return;
  }

  // Build word-level timeline if available
  let words = [];
  if (Array.isArray(chunks) && chunks.length) {
    // Transformers.js whisper chunks often look like:
    // { text: "...", timestamp: [startSec, endSec] } for word mode
    for (const c of chunks) {
      const ts = c.timestamp;
      if (!ts || ts.length < 2) continue;
      const startSec = ts[0] ?? 0;
      const endSec = ts[1] ?? startSec;
      const w0 = winStart + Math.floor(startSec * TARGET_SR);
      const w1 = winStart + Math.floor(endSec * TARGET_SR);
      const e0 = sampleToEpochMs(w0);
      const e1 = sampleToEpochMs(w1);
      if (e0 == null || e1 == null) continue;

      const wText = (c.text ?? "").trim();
      if (!wText) continue;

      words.push({ t0: e0, t1: e1, text: wText, speakerId: null });
    }
  }

  // If diarization enabled, assign speakers to words by overlap with diarization segments
  if (VAD.enabled) {
    assignSpeakersToWords(words);
  }

  // Add to global structures
  if (words.length) transcriptWords.push(...words);

  transcriptSegments.push({
    t0: t0Epoch,
    t1: t1Epoch,
    text: text.trim(),
    rawText: text.trim(),
    words: words.length ? words : null,
    latencyMs: ms,
  });

  renderTranscript();
  logDiag(`ASR window ${formatRelTime(t0Epoch)} → ${formatRelTime(t1Epoch)} (${ms}ms), words=${words.length}`);

  if (autoRunSel.value === "on") maybeTranscribe();
  else btnTranscribe.disabled = !readWindowFromBlocks(getWindowParams().winSamples);
}

function renderTranscript() {
  const showRaw = showRawChk.checked;

  // Prefer word-level rendering when available and timestamps on.
  if (!showRaw && transcriptWords.length && timestampsSel.value !== "off") {
    // Render with speaker and timestamps.
    // We group words into lines by time gaps or speaker changes.
    const lines = [];
    let current = null;

    for (const w of transcriptWords) {
      const spk = w.speakerId ? `S${w.speakerId}` : "S?";
      const ts = formatRelTime(w.t0);

      if (!current) {
        current = { spk, ts, tLast: w.t1, text: w.text };
        continue;
      }

      const gapMs = (w.t0 - current.tLast);
      const speakerChanged = spk !== current.spk;
      if (gapMs > 1200 || speakerChanged) {
        lines.push(`${current.ts} ${current.spk}: ${current.text}`);
        current = { spk, ts, tLast: w.t1, text: w.text };
      } else {
        current.text += (needsSpace(current.text, w.text) ? " " : "") + w.text;
        current.tLast = w.t1;
      }
    }
    if (current) lines.push(`${current.ts} ${current.spk}: ${current.text}`);

    transcriptTa.value = lines.join("\n");
  } else {
    // Segment-level rendering
    const lines = transcriptSegments.map(seg => {
      const t = seg.t0 != null ? formatRelTime(seg.t0) : "[--:--:--.---]";
      return `${t} ${seg.text}`;
    });
    transcriptTa.value = lines.join("\n");
  }

  if (autoScrollChk.checked) {
    transcriptTa.scrollTop = transcriptTa.scrollHeight;
  }
}

function needsSpace(a, b) {
  if (!a || !b) return true;
  const last = a[a.length - 1];
  const first = b[0];
  if (/\s/.test(last)) return false;
  if (/[\(\[\{]/.test(first)) return false;
  if (/[.,!?;:\)\]\}]/.test(first)) return false;
  return true;
}

// ----------- Diarization (VAD + clustering) ------------

// Observe each worklet chunk, create VAD segments
function vadObserveChunk(startSample, length, rms) {
  const now = Date.now();

  // simple RMS threshold
  const isSpeech = rms >= VAD.rmsThreshold;

  if (isSpeech) {
    VAD.lastSpeechMs = now;
    if (!VAD.speechOn) {
      VAD.speechOn = true;
      VAD.currentSeg = {
        startSample,
        endSample: startSample + length,
        blocks: [{ startSample, length }],
      };
    } else if (VAD.currentSeg) {
      VAD.currentSeg.endSample = startSample + length;
      VAD.currentSeg.blocks.push({ startSample, length });
    }
  } else {
    if (VAD.speechOn) {
      // hangover logic
      if ((now - VAD.lastSpeechMs) > VAD.hangoverMs) {
        finalizeVadSegmentIfAny(false);
        VAD.speechOn = false;
        VAD.currentSeg = null;
      }
    }
  }

  // cap long segments
  if (VAD.speechOn && VAD.currentSeg) {
    const segMs = ((VAD.currentSeg.endSample - VAD.currentSeg.startSample) / TARGET_SR) * 1000;
    if (segMs > VAD.maxSegMs) {
      finalizeVadSegmentIfAny(false);
      VAD.speechOn = false;
      VAD.currentSeg = null;
    }
  }
}

function finalizeVadSegmentIfAny(force) {
  if (!VAD.currentSeg) return;

  const seg = VAD.currentSeg;
  const durMs = ((seg.endSample - seg.startSample) / TARGET_SR) * 1000;

  if (!force && durMs < VAD.minSegMs) {
    // ignore
    VAD.currentSeg = null;
    return;
  }

  const pcm = readRangeFromBlocks(seg.startSample, seg.endSample);
  if (!pcm) {
    VAD.currentSeg = null;
    return;
  }

  const t0 = sampleToEpochMs(seg.startSample);
  const t1 = sampleToEpochMs(seg.endSample);

  VAD.segments.push({
    id: Date.now() + Math.random(),
    t0, t1,
    startSample: seg.startSample,
    endSample: seg.endSample,
    pcm,
    speakerId: null,
  });

  // Kick off embedding -> speaker assign
  requestSpeakerEmbedding(VAD.segments[VAD.segments.length - 1]);

  VAD.currentSeg = null;
}

function readRangeFromBlocks(startSample, endSample) {
  const n = endSample - startSample;
  if (n <= 0) return null;

  const out = new Float32Array(n);

  let filled = 0;
  for (const b of pcmBlocks) {
    const b0 = b.startSample;
    const b1 = b.startSample + b.pcm.length;
    if (b1 <= startSample) continue;
    if (b0 >= endSample) break;

    const copyStart = Math.max(b0, startSample);
    const copyEnd = Math.min(b1, endSample);
    const srcOffset = copyStart - b0;
    const dstOffset = copyStart - startSample;
    const len = copyEnd - copyStart;

    out.set(b.pcm.subarray(srcOffset, srcOffset + len), dstOffset);
    filled += len;
    if (filled >= n) break;
  }

  return out;
}

function requestSpeakerEmbedding(seg) {
  if (!VAD.enabled) return;
  if (!spkReady) return;

  // Downstream: worker returns embedding; we cluster and set seg.speakerId.
  worker.postMessage({
    type: "embed",
    id: seg.id,
    audio: seg.pcm,
    samplingRate: TARGET_SR,
  }, [seg.pcm.buffer]);

  setDiaPill("running…", "warn");
}

function handleSpeakerEmbedding({ id, embedding, ms }) {
  const seg = VAD.segments.find(s => s.id === id);
  if (!seg) return;

  const threshold = clamp(Number(spkThreshIn.value) || 0.78, 0.5, 0.95);
  const assigned = assignToSpeakerCluster(embedding, threshold);

  seg.speakerId = assigned;
  setDiaPill(`ready (${spkDevice})`, "good");
  logDiag(`SPK embed ${id}: speaker=S${assigned} (${ms}ms)`);

  // Re-assign speakers for any words overlapping this segment
  // (cheap incremental update)
  if (transcriptWords.length) {
    for (const w of transcriptWords) {
      if (w.t1 <= seg.t0 || w.t0 >= seg.t1) continue;
      w.speakerId = assigned;
    }
    renderTranscript();
  }
}

function assignToSpeakerCluster(vec, threshold) {
  // cosine similarity since vectors are l2-normalized
  let bestId = null;
  let bestSim = -1;

  for (const c of speakerClusters) {
    const sim = dot(vec, c.centroid);
    if (sim > bestSim) { bestSim = sim; bestId = c.id; }
  }

  if (bestId == null || bestSim < threshold) {
    // new speaker
    const id = nextSpeakerId++;
    speakerClusters.push({ id, centroid: vec.slice(0), count: 1 });
    return id;
  }

  // update centroid (online average)
  const cl = speakerClusters.find(c => c.id === bestId);
  cl.count += 1;
  for (let i = 0; i < cl.centroid.length; i++) {
    cl.centroid[i] = cl.centroid[i] + (vec[i] - cl.centroid[i]) / cl.count;
  }
  // re-normalize
  l2Normalize(cl.centroid);

  return bestId;
}

function dot(a, b) {
  const n = Math.min(a.length, b.length);
  let s = 0;
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}
function l2Normalize(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  const n = Math.sqrt(s) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= n;
}

// Assign speakers to newly emitted ASR words using overlap with known VAD speaker segments.
function assignSpeakersToWords(words) {
  if (!VAD.enabled) return;
  if (!VAD.segments.length) return;

  for (const w of words) {
    const spk = speakerForTimeRange(w.t0, w.t1);
    if (spk != null) w.speakerId = spk;
  }
}

function speakerForTimeRange(t0, t1) {
  // Choose the segment with max overlap that has a speakerId
  let best = null;
  let bestOverlap = 0;

  for (const s of VAD.segments) {
    if (!s.speakerId) continue;
    if (s.t1 <= t0 || s.t0 >= t1) continue;
    const overlap = Math.min(s.t1, t1) - Math.max(s.t0, t0);
    if (overlap > bestOverlap) {
      bestOverlap = overlap;
      best = s.speakerId;
    }
  }
  return best;
}

// ----------- OpenWebUI (same as prior version) ------------

function getOpenWebUIConfig() {
  const baseUrl = owBaseUrlIn.value.trim().replace(/\/+$/, "");
  const apiKey = owApiKeyIn.value;
  const model = owModelSel.value;
  const stream = owStreamSel.value === "on";
  if (!baseUrl) throw new Error("OpenWebUI Base URL is required.");
  if (!apiKey) throw new Error("OpenWebUI API key is required (not stored).");
  return { baseUrl, apiKey, model, stream };
}

async function owFetch(path, { baseUrl, apiKey }, init = {}) {
  const url = `${baseUrl}${path}`;
  const headers = new Headers(init.headers || {});
  headers.set("Authorization", `Bearer ${apiKey}`);
  return fetch(url, { ...init, headers });
}

async function loadOpenWebUIModels() {
  try {
    const cfg = getOpenWebUIConfig();
    setOwuiPill("loading models…", "warn");

    const res = await owFetch("/api/models", cfg, { method: "GET" });
    if (!res.ok) throw new Error(`OpenWebUI /api/models ${res.status}: ${await res.text()}`);

    const data = await res.json();
    const list = Array.isArray(data?.data) ? data.data : (Array.isArray(data) ? data : []);

    owModelSel.innerHTML = "";
    const ph = document.createElement("option");
    ph.value = "";
    ph.textContent = list.length ? "(select model)" : "(no models returned)";
    owModelSel.appendChild(ph);

    for (const m of list) {
      const opt = document.createElement("option");
      opt.value = m.id || m.model || "";
      opt.textContent = opt.value || JSON.stringify(m);
      owModelSel.appendChild(opt);
    }

    setOwuiPill("models loaded", "good");
    logDiag(`Loaded ${list.length} OpenWebUI models.`);
    persistSettings();
  } catch (e) {
    setOwuiPill("error (models)", "bad");
    logDiag(`Load models error: ${String(e)}`);
  }
}

async function testOpenWebUI() {
  try {
    const cfg = getOpenWebUIConfig();
    setOwuiPill("testing…", "warn");

    const res = await owFetch("/api/models", cfg, { method: "GET" });
    if (!res.ok) throw new Error(`Test failed ${res.status}: ${await res.text()}`);

    setOwuiPill("connected", "good");
    logDiag("OpenWebUI test OK.");
  } catch (e) {
    setOwuiPill("CORS/auth error", "bad");
    logDiag(`OpenWebUI test error: ${String(e)}`);
  }
}

function buildSummaryPrompt(style) {
  if (style === "bullets") {
    return { system: "Summarize as concise bullets. No fluff.", userPrefix: "Summarize this transcript delta:" };
  }
  if (style === "actions") {
    return { system: "Extract action items, decisions, and open questions. Be concise.", userPrefix: "Extract from this transcript delta:" };
  }
  if (style === "json") {
    return {
      system: "Return ONLY valid JSON. Schema: {summary:string,key_points:string[],action_items:[{item:string,owner:string|null}],decisions:string[],risks:string[],questions:string[]}.",
      userPrefix: "Convert this transcript delta to the JSON schema:"
    };
  }
  return {
    system: "Write succinct meeting notes with headings: Summary, Key Points, Action Items, Decisions, Questions.",
    userPrefix: "Update the running notes with this transcript delta:"
  };
}

function getTranscriptDelta() {
  const full = transcriptTa.value || "";
  if (full.length <= lastSummarizedChar) return "";
  return full.slice(lastSummarizedChar).trim();
}

function startSummaryTimer() {
  stopSummaryTimer();
  summaryTimer = setInterval(async () => {
    try { await summarizeIfNeeded(); } catch (e) { logDiag(`Auto-summary error: ${String(e)}`); }
  }, 1000);
}
function stopSummaryTimer() {
  if (summaryTimer) clearInterval(summaryTimer);
  summaryTimer = null;
}

async function summarizeIfNeeded() {
  const every = clamp(Number(sumEverySecIn.value) || 60, 15, 600);
  const now = Date.now();
  if (!summarizeIfNeeded._last) summarizeIfNeeded._last = 0;
  if (now - summarizeIfNeeded._last < every * 1000) return;

  const delta = getTranscriptDelta();
  if (delta.length < 250) return;

  summarizeIfNeeded._last = now;
  await summarizeNow(true);
}

async function summarizeNow(isAuto) {
  try {
    const cfg = getOpenWebUIConfig();
    if (!cfg.model) throw new Error("Select an OpenWebUI model first (Load models).");

    const delta = getTranscriptDelta();
    if (!delta) { logDiag("No new transcript text to summarize."); return; }

    setOwuiPill(isAuto ? "auto summarizing…" : "summarizing…", "warn");

    const prompt = buildSummaryPrompt(sumStyleSel.value);
    const existing = (summaryTa.value || "").trim();
    const running = existing
      ? `Current running summary:\n\n${existing}\n\nUpdate it with new transcript delta below.`
      : `Create the initial summary from the transcript delta below.`;

    const messages = [
      { role: "system", content: prompt.system },
      { role: "user", content: `${running}\n\n${prompt.userPrefix}\n\n${delta}` },
    ];

    if (cfg.stream) {
      await openWebUIChatStream(cfg, messages, (partial) => { summaryTa.value = partial; });
    } else {
      const json = await openWebUIChat(cfg, messages);
      summaryTa.value = (json?.choices?.[0]?.message?.content ?? "").trim();
    }

    lastSummarizedChar = (transcriptTa.value || "").length;
    setOwuiPill("connected", "good");
    logDiag(`Summarized delta chars=${delta.length}`);
  } catch (e) {
    setOwuiPill("error", "bad");
    logDiag(`Summarize error: ${String(e)}`);
  }
}

async function openWebUIChat({ baseUrl, apiKey, model }, messages) {
  const url = `${baseUrl.replace(/\/+$/, "")}/api/chat/completions`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Authorization": `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify({ model, messages, temperature: 0.2, stream: false }),
  });
  if (!res.ok) throw new Error(`OpenWebUI ${res.status}: ${await res.text()}`);
  return res.json();
}

async function openWebUIChatStream({ baseUrl, apiKey, model }, messages, onText) {
  const url = `${baseUrl.replace(/\/+$/, "")}/api/chat/completions`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Authorization": `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify({ model, messages, temperature: 0.2, stream: true }),
  });
  if (!res.ok) throw new Error(`OpenWebUI stream ${res.status}: ${await res.text()}`);
  if (!res.body) throw new Error("Streaming response has no body.");

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");

  let buf = "";
  let assembled = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop() || "";

    for (const part of parts) {
      const lines = part.split("\n").map(l => l.trim());
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (!data || data === "[DONE]") continue;

        try {
          const json = JSON.parse(data);
          const delta = json?.choices?.[0]?.delta?.content
                     ?? json?.choices?.[0]?.message?.content
                     ?? "";
          if (delta) { assembled += delta; onText(assembled); }
        } catch {}
      }
    }
  }
  onText(assembled.trim());
}

// ----------- Export ------------

function exportJson() {
  const obj = {
    createdAt: new Date().toISOString(),
    captureStartEpochMs,
    settings: {
      asrModel: asrModelSel.value,
      devicePref: devicePrefSel.value,
      windowSec: Number(winSecIn.value) || 15,
      overlapSec: Number(ovlSecIn.value) || 3,
      timestamps: timestampsSel.value,
      diarization: diarizeSel.value,
      speakerThreshold: Number(spkThreshIn.value) || 0.78,
      openWebUIBaseUrl: owBaseUrlIn.value.trim(),
      openWebUIModel: owModelSel.value,
      summaryStyle: sumStyleSel.value,
    },
    transcriptText: transcriptTa.value || "",
    summaryText: summaryTa.value || "",
    transcriptSegments,
    transcriptWords,
    diarizationSegments: VAD.segments.map(s => ({
      t0: s.t0, t1: s.t1,
      startSample: s.startSample, endSample: s.endSample,
      speakerId: s.speakerId ?? null
    })),
    speakers: speakerClusters.map(c => ({ id: c.id, count: c.count })),
  };
  downloadBlob(JSON.stringify(obj, null, 2), "application/json", `transcript_${stamp()}.json`);
}

function exportTxt() {
  const text =
`Transcript
==========
${transcriptTa.value || ""}

Summary
=======
${summaryTa.value || ""}

Generated: ${new Date().toISOString()}
`;
  downloadBlob(text, "text/plain", `transcript_${stamp()}.txt`);
}

function stamp() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${pad(d.getMonth()+1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

function downloadBlob(text, mime, filename) {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
