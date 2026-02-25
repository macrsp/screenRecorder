// Worker that hosts:
// - Whisper ASR pipeline (Transformers.js)
// - Optional speaker embedding pipeline for diarization (feature-extraction)
//
// Messages in:
// - {type:"init_asr", model, devicePref}
// - {type:"init_spk", model, devicePref}
// - {type:"transcribe", id, audio:Float32Array, samplingRate, timestamps}
// - {type:"embed", id, audio:Float32Array, samplingRate}
//
// Messages out:
// - asr_ready, asr_status, asr_result, asr_error
// - spk_ready, spk_status, spk_embed, spk_error

let asr = null;
let spk = null;

let asrBusy = false;
let spkBusy = false;

let asrModel = null;
let spkModel = null;
let deviceInUse = null;

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

async function loadTransformers() {
  return import("https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js");
}

function pickDevice(devicePref) {
  if (devicePref === "wasm") return "wasm";
  if (devicePref === "webgpu") return "webgpu";
  // auto
  return "webgpu";
}

async function initASR({ model, devicePref }) {
  post("asr_status", { status: "loading" });

  const { pipeline, env } = await loadTransformers();
  env.useBrowserCache = true;

  let device = pickDevice(devicePref);
  try {
    asr = await pipeline("automatic-speech-recognition", model, { device });
  } catch (e) {
    if (device !== "wasm") {
      post("asr_status", { status: "webgpu_failed_fallback_wasm", error: String(e) });
      device = "wasm";
      asr = await pipeline("automatic-speech-recognition", model, { device });
    } else {
      throw e;
    }
  }

  asrModel = model;
  deviceInUse = device;

  post("asr_ready", { model: asrModel, device });
  post("asr_status", { status: "ready" });
}

// Speaker embedding model: choose a compact one.
// You may want to change this to a model you’ve tested for speaker verification.
// Default here is conservative: users can swap model in code if desired.
async function initSPK({ model, devicePref }) {
  post("spk_status", { status: "loading" });

  const { pipeline, env } = await loadTransformers();
  env.useBrowserCache = true;

  let device = pickDevice(devicePref);
  try {
    spk = await pipeline("feature-extraction", model, { device });
  } catch (e) {
    if (device !== "wasm") {
      post("spk_status", { status: "webgpu_failed_fallback_wasm", error: String(e) });
      device = "wasm";
      spk = await pipeline("feature-extraction", model, { device });
    } else {
      throw e;
    }
  }

  spkModel = model;
  post("spk_ready", { model: spkModel, device });
  post("spk_status", { status: "ready" });
}

async function transcribe({ id, audio, samplingRate, timestamps }) {
  if (!asr) throw new Error("ASR not initialized");
  if (asrBusy) {
    post("asr_busy", { id });
    return;
  }
  asrBusy = true;

  const t0 = performance.now();
  const return_timestamps = (timestamps === "off") ? false : timestamps;

  try {
    const pcm = (audio instanceof Float32Array) ? audio : new Float32Array(audio);
    const result = await asr(pcm, {
      sampling_rate: samplingRate ?? 16000,
      return_timestamps,
    });

    const t1 = performance.now();
    const text = (result?.text ?? "").trim();
    const chunks = Array.isArray(result?.chunks) ? result.chunks : null;

    post("asr_result", { id, text, chunks, ms: Math.round(t1 - t0) });
  } catch (e) {
    post("asr_error", { id, error: String(e) });
  } finally {
    asrBusy = false;
  }
}

// Returns a single vector embedding.
// We pool mean across time if needed.
async function embed({ id, audio, samplingRate }) {
  if (!spk) throw new Error("Speaker embedding not initialized");
  if (spkBusy) {
    post("spk_busy", { id });
    return;
  }
  spkBusy = true;

  const t0 = performance.now();
  try {
    const pcm = (audio instanceof Float32Array) ? audio : new Float32Array(audio);

    // feature-extraction typically returns [1, T, D] or [T, D] depending on model
    const out = await spk(pcm, { sampling_rate: samplingRate ?? 16000 });

    // Normalize to a flat vector by mean pooling over time
    const vec = meanPoolToVector(out);
    l2Normalize(vec);

    const t1 = performance.now();
    post("spk_embed", { id, embedding: vec, ms: Math.round(t1 - t0) });
  } catch (e) {
    post("spk_error", { id, error: String(e) });
  } finally {
    spkBusy = false;
  }
}

function meanPoolToVector(out) {
  // Try common shapes
  // out might be a Tensor-like with .data / .dims in Transformers.js
  if (out && out.data && out.dims) {
    const { data, dims } = out;
    // dims could be [1, T, D] or [T, D] or [1, D]
    if (dims.length === 3) {
      const T = dims[1], D = dims[2];
      const v = new Float32Array(D);
      let idx = 0;
      // data length = 1*T*D
      for (let t = 0; t < T; t++) {
        for (let d = 0; d < D; d++) v[d] += data[idx++];
      }
      for (let d = 0; d < D; d++) v[d] /= Math.max(1, T);
      return v;
    }
    if (dims.length === 2) {
      const T = dims[0], D = dims[1];
      const v = new Float32Array(D);
      let idx = 0;
      for (let t = 0; t < T; t++) {
        for (let d = 0; d < D; d++) v[d] += data[idx++];
      }
      for (let d = 0; d < D; d++) v[d] /= Math.max(1, T);
      return v;
    }
    if (dims.length === 1) {
      return Float32Array.from(data);
    }
  }

  // Fallback: if it’s already an array
  if (Array.isArray(out)) {
    if (Array.isArray(out[0])) {
      // [T][D]
      const T = out.length;
      const D = out[0].length;
      const v = new Float32Array(D);
      for (let t = 0; t < T; t++) for (let d = 0; d < D; d++) v[d] += out[t][d];
      for (let d = 0; d < D; d++) v[d] /= Math.max(1, T);
      return v;
    }
    return Float32Array.from(out);
  }

  throw new Error("Unexpected embedding output shape");
}

function l2Normalize(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  const n = Math.sqrt(s) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= n;
}

self.onmessage = async (e) => {
  const msg = e.data || {};
  try {
    if (msg.type === "init_asr") await initASR(msg);
    else if (msg.type === "init_spk") await initSPK(msg);
    else if (msg.type === "transcribe") await transcribe(msg);
    else if (msg.type === "embed") await embed(msg);
  } catch (err) {
    post("asr_error", { error: String(err) });
  }
};
