# Local Screen Audio Transcriber
## Browser Whisper (Transformers.js) + Sample-Accurate Timestamp Sync + Best-Effort Speaker Diarization + OpenWebUI Summaries

This repository contains a **static web application** that:

- Captures **system audio** from **screen/tab/window capture** (`getDisplayMedia`)
- Extracts **16 kHz mono PCM** in real time using an **AudioWorklet**
- Runs **Whisper ASR locally in the browser** (Transformers.js) to produce a transcript
- Produces **tight, sample-accurate timestamp synchronization** across the entire capture session
- Optionally performs **best-effort speaker diarization** locally (no server required)
- Sends **only transcript text** (not audio) to **OpenWebUI** for summarization (user supplies OpenWebUI API key)

### Primary constraints this design satisfies

- **No audio upload**: audio stays on the client. No backend sees audio.
- **Static hosting**: can be served from any static host (including IIS).
- **No secrets on server**: OpenWebUI API key is entered by the user and kept in memory only.
- **Works with existing OpenWebUI user base**: avoids requiring Azure API keys per user.

---

## Table of Contents

1. [What you get](#what-you-get)
2. [High-level architecture](#high-level-architecture)
3. [How capture works](#how-capture-works)
4. [Timestamp synchronization model](#timestamp-synchronization-model)
5. [Transcription pipeline (Whisper in browser)](#transcription-pipeline-whisper-in-browser)
6. [Speaker diarization (best-effort local)](#speaker-diarization-best-effort-local)
7. [Summaries via OpenWebUI](#summaries-via-openwebui)
8. [Repository layout](#repository-layout)
9. [Quick start](#quick-start)
10. [Configuration reference](#configuration-reference)
11. [Hosting and deployment](#hosting-and-deployment)
    - [Any static host](#any-static-host)
    - [IIS hosting](#iis-hosting)
    - [IIS reverse proxy to OpenWebUI (same-origin)](#iis-reverse-proxy-to-openwebui-same-origin)
12. [CORS guidance and patterns](#cors-guidance-and-patterns)
13. [Security guidance](#security-guidance)
14. [Performance guidance](#performance-guidance)
15. [Troubleshooting](#troubleshooting)
16. [Exports and integration](#exports-and-integration)
17. [Extending the app](#extending-the-app)
18. [FAQ](#faq)
19. [Operational checklists](#operational-checklists)
20. [License](#license)

---

## What you get

### Transcription
- Whisper runs **in-browser** via Transformers.js.
- Incremental transcript generation in **windowed** chunks (e.g., 15s windows with 3s overlap).
- Optional **word timestamps** (recommended for tight sync).

### Timestamp sync
- **Sample-index-based timeline** for the entire capture session.
- Converts sample index → absolute epoch ms → relative display time `[HH:MM:SS.mmm]`.
- Whisper timestamps (relative to window) are mapped into the global timeline.

### Speaker diarization (optional)
- Best-effort diarization using:
  - Naive VAD (RMS threshold + hangover)
  - Speaker embeddings via Transformers.js `feature-extraction`
  - Online cosine-similarity clustering
  - Word assignment by time overlap with diarization segments

### Summaries (OpenWebUI)
- Summaries computed from transcript delta using OpenWebUI `/api/chat/completions`.
- Supports streaming (SSE) and non-streaming mode.
- Supports model listing via OpenWebUI `/api/models`.

---

## High-level architecture

### Data flows (no audio upload)

```

┌─────────────────────────────────────────────────────────────────────┐
│ Browser                                                            │
│                                                                     │
│  getDisplayMedia()                                                  │
│     │                                                               │
│     ▼                                                               │
│  MediaStream (video+audio)                                          │
│     │                                                               │
│     ▼                                                               │
│  AudioContext + AudioWorklet                                        │
│     │  (downsample to 16kHz mono + sample index)                    │
│     ▼                                                               │
│  PCM blocks with absolute sample indices                            │
│     │                                                               │
│     ├────────► ASR Worker (Transformers.js Whisper) ────────┐       │
│     │                                                       │       │
│     ├────────► (optional) Speaker embedding Worker ───────┐ │       │
│     │                                                    │ │       │
│     ▼                                                    ▼ ▼       │
│  Timestamped words (t0,t1,text)                    Speaker clusters  │
│     │                                                    │          │
│     └────────► Speaker assignment to words ◄──────────────┘          │
│     │                                                               │
│     ▼                                                               │
│  Transcript UI + structured export (.json)                           │
│     │                                                               │
│     ▼                                                               │
│  Summaries: fetch to OpenWebUI (user API key)                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

```

### Network boundaries

- **Audio**: never leaves browser.
- **Text**: transcript deltas are sent to OpenWebUI if summarization is enabled.
- **Keys**: OpenWebUI key entered by user in UI; kept in memory only.

---

## How capture works

### Capture API
The app uses:

- `navigator.mediaDevices.getDisplayMedia({ video: true, audio: true })`

Important implications:
- Requires a **secure context**: `https://` or `http://localhost`.
- On Windows/Edge, capturing **system audio** requires the user to select a surface that supports “Share audio”.

### PCM extraction
MediaRecorder typically produces compressed WebM/Opus; Whisper expects PCM.
Therefore the app uses Web Audio:

- `AudioContext`
- `createMediaStreamSource(stream)`
- `AudioWorkletNode`

The AudioWorklet:
- mixes stereo to mono (if needed)
- downsamples to 16 kHz (linear interpolation baseline)
- emits PCM chunks plus an **absolute sample index counter** at 16 kHz

---

## Timestamp synchronization model

This is the core of “full timestamp synchronization”.

### Timeline definitions

- `captureStartEpochMs`: wall-clock time when user clicks “Start capture” and stream is obtained.
- `firstSampleStart`: the start sample index of the first PCM chunk received from the AudioWorklet. Usually 0, but treat it as unknown and store it.
- `TARGET_SR = 16000`.

### Sample index → epoch ms

For a given absolute sample index `s`:

```

relSamples = s - firstSampleStart
relMs = (relSamples / TARGET_SR) * 1000
epochMs = captureStartEpochMs + relMs

```

### Whisper timestamps mapping
Whisper returns timestamps relative to the audio buffer you send (the ASR window).
If a word has `[startSec, endSec]` relative to the window:

```

wordStartSample = windowStartSample + startSec * TARGET_SR
wordEndSample   = windowStartSample + endSec   * TARGET_SR
wordStartEpochMs = sampleToEpochMs(wordStartSample)
wordEndEpochMs   = sampleToEpochMs(wordEndSample)

```

### Display format
The UI commonly displays relative time:

- `epochMs - captureStartEpochMs` formatted as `[HH:MM:SS.mmm]`.

### Why this is “full sync”
Because:
- Every audio chunk has an absolute sample index
- Every ASR window has an absolute sample range
- Every word timestamp is derived from the sample clock
- All timestamps align to the same global session clock

This remains consistent even if:
- inference latency varies
- windows overlap
- diarization runs slower/faster than ASR

---

## Transcription pipeline (Whisper in browser)

### Overview
Whisper is not truly streaming. The common “real-time” pattern is:

- Keep a ring buffer
- Extract a window (e.g., 15 seconds)
- Transcribe
- Keep an overlap (e.g., 3 seconds)
- Repeat

### Why overlap exists
Overlap reduces boundary artifacts:
- a word starting at the end of a window might be truncated
- overlap reintroduces that context

Overlap can introduce duplication:
- you can post-process using timestamps (recommended), or accept minor repetition in plain text view.

### Worker-based inference
Inference is run in a dedicated Web Worker to:
- keep UI responsive
- keep AudioWorklet message handling smooth
- avoid main thread jank (which can cause audio glitches)

### WebGPU vs WASM
The app supports:
- WebGPU backend (fast) when available
- WASM backend fallback (slower)

---

## Speaker diarization (best-effort local)

### What diarization means here
Given audio with multiple speakers, diarization aims to label segments as:
- Speaker 1, Speaker 2, etc.

### What you should expect
This implementation is “best-effort” because:
- Dedicated diarization models/pipelines (e.g., pyannote) are not trivially portable to static web-only environments.
- Overlapping speech and low-quality audio significantly degrade performance.

### Approach used in this repo
1. **Speech segmentation (naive VAD)**
   - Uses RMS threshold on PCM chunks to decide speech/not-speech.
   - Uses hangover time to avoid chopping speech too aggressively.
   - Produces speech segments `(t0,t1)` in global timebase.

2. **Speaker embedding**
   - For each speech segment, compute an embedding using a feature extraction model in Transformers.js.
   - Normalize embedding with L2 norm.

3. **Online clustering**
   - Maintain speaker clusters with centroids.
   - Assign segment to nearest centroid by cosine similarity:
     - if similarity ≥ threshold: same speaker
     - else: new speaker cluster
   - Update centroid via running mean.

4. **Word alignment**
   - Whisper words have `(t0,t1)`.
   - A word is assigned to the speaker whose diarization segment overlaps most.

### Common tuning knobs
- Speaker similarity threshold (0.75–0.85 typical).
- VAD RMS threshold: controls sensitivity.

### When to disable diarization
- If you only need a single transcript stream.
- If users run on CPU-only devices (WASM) and performance is unacceptable.

---

## Summaries via OpenWebUI

### Why OpenWebUI
- You have a large existing OpenWebUI user base with API keys enabled.
- Avoid requiring users to have their own Azure API key.

### API usage
The UI uses OpenWebUI’s OpenAI-compatible endpoints:

- `GET /api/models` to list models
- `POST /api/chat/completions` for summarization

### Key handling
- API key is entered in the UI
- Stored in memory only
- Not written to LocalStorage or disk

### Delta summarization
To control token usage and avoid re-sending the entire transcript repeatedly:

- Track `lastSummarizedChar`
- Each summary call sends the transcript substring from that position to end (“delta”)
- The prompt includes current summary to “update the running notes”

---

## Repository layout

```

/app
index.html        UI + controls
styles.css        styling
main.js           main thread logic (capture, buffering, timestamp mapping, diarization orchestration, OpenWebUI calls)
pcm-worklet.js    AudioWorklet for PCM extraction + downsampling + absolute sample index
asr-worker.js     Web Worker hosting ASR pipeline (Whisper) + optional speaker embedding pipeline
README.md

````

---

## Quick start

### 1) Serve locally

#### Python
```bash
python3 -m http.server 8080
````

Open:

* `http://localhost:8080/app/`

#### Node

```bash
npx serve .
```

Open the printed URL, navigate to `/app/`.

> `getDisplayMedia()` requires HTTPS unless on `localhost`.

### 2) Capture and transcribe

1. In **Capture**:

   * ASR model: start with `Xenova/whisper-base.en`
   * Word timestamps: set to `word` (recommended)
   * Device preference: `auto`
2. Click **Start capture**
3. In the picker:

   * Choose tab/window/screen
   * Enable **Share audio** if available
4. Transcript begins to appear

### 3) Enable diarization (optional)

* Set **Enable diarization** to `on`
* Start capture
* If speakers merge/split incorrectly, adjust threshold

### 4) Summarize via OpenWebUI

1. Fill:

   * Base URL: `https://your-openwebui-host`
   * API key: paste user key
2. Click **Load models**, pick a model
3. Click **Summarize now** (or use periodic summaries)

---

## Configuration reference

### Capture panel

#### ASR model

* `Xenova/whisper-tiny.en`: fastest, lowest accuracy
* `Xenova/whisper-base.en`: recommended baseline
* `Xenova/whisper-small.en`: higher accuracy, heavier

#### Device preference

* `auto`: prefer WebGPU, fallback to WASM
* `webgpu`: force WebGPU
* `wasm`: force CPU

#### Window seconds

* Typical: 10–20 seconds
* Larger windows: better context, higher latency

#### Overlap seconds

* Typical: 2–4 seconds
* Higher overlap: better boundary behavior, more duplication

#### Word timestamps

* `word`: best for alignment and diarization
* `chunk`: less granular
* `off`: fastest

#### Enable diarization

* `off`: no speaker labels
* `on`: enable speaker labels

#### Speaker similarity threshold

* Higher threshold: merges speakers more (fewer speakers)
* Lower threshold: splits more (more speakers)
* Start: `0.78`
* Typical: `0.75–0.85`

### OpenWebUI panel

#### Base URL

Examples:

* `https://openwebui.company.com`
* `https://example.com/openwebui` (if reverse proxied under a path)

#### API key

* Must be a valid OpenWebUI API key
* Not stored

#### Model

* Populated from `/api/models`

#### Summary interval

* Runs periodic delta summaries

#### Summary style

* meeting notes / bullets / action items / JSON

#### Streaming

* `on`: SSE streaming (lower perceived latency)
* `off`: non-streaming (more robust through proxies)

---

## Hosting and deployment

### Any static host

You can host `/app` on:

* Nginx / Apache
* S3 + CloudFront
* GitHub Pages
* Azure Static Web Apps
* IIS
* Any CDN-backed static bucket

Requirements:

* Serve over HTTPS (unless localhost)
* Correct MIME types for JavaScript files
* Ensure the app path resolves worker scripts correctly (relative paths are used)

---

## IIS hosting

Yes—this app can be hosted on **IIS**. It is a static web application.

### Step 1: Copy files to IIS web root

Example:

* `C:\inetpub\wwwroot\local-transcriber\app\`

You can either:

* Host the `/app` folder directly as the site root, OR
* Host the repository root and navigate to `/app/` in the browser

Recommendation:

* Make the site root point to the `/app` folder to keep URLs simple.

### Step 2: Create a site (or virtual directory)

#### Option A: New Website

In IIS Manager:

1. Right click **Sites** → **Add Website…**
2. Site name: `LocalTranscriber`
3. Physical path: `C:\inetpub\wwwroot\local-transcriber\app`
4. Binding: choose host name and port

#### Option B: Virtual Directory under an existing site

1. Select your site
2. Right click → **Add Virtual Directory…**
3. Alias: `transcriber`
4. Physical path: `...\local-transcriber\app`

Then the URL is:

* `https://yourhost/transcriber/`

### Step 3: Enable HTTPS

`getDisplayMedia()` requires HTTPS except on localhost.

In IIS Manager:

1. Select site → **Bindings…**
2. Add `https`
3. Pick certificate

### Step 4: Configure MIME types (required)

For ES modules and worker scripts, `.js` must be served with a valid JS MIME type.

In IIS Manager:

1. Select site → **MIME Types**
2. Add:

   * Extension: `.js`
   * MIME type: `text/javascript`

Optional (if you later add these):

* `.mjs` → `text/javascript`
* `.wasm` → `application/wasm`

### Step 5: Optional web.config (MIME types)

Create `web.config` in the same folder as `index.html`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <system.webServer>
    <staticContent>
      <mimeMap fileExtension=".js" mimeType="text/javascript" />
      <mimeMap fileExtension=".mjs" mimeType="text/javascript" />
      <mimeMap fileExtension=".wasm" mimeType="application/wasm" />
    </staticContent>
  </system.webServer>
</configuration>
```

**Important**: If IIS already has `.js` registered, adding it again can cause a duplicate entry error. In that case:

* remove the duplicate line, or
* configure it at the server level instead of site level

### Step 6: Verify

* Open the site over HTTPS in Edge
* Confirm `main.js`, `asr-worker.js`, and `pcm-worklet.js` load (DevTools → Network)
* Confirm the “Start capture” button works and the share picker appears

---

## IIS reverse proxy to OpenWebUI (same-origin)

If you want to eliminate CORS problems for OpenWebUI calls, the cleanest pattern is:

* Serve the transcriber UI and OpenWebUI from the **same origin**
* Use reverse proxy/path routing

### Why same-origin is best

* No CORS preflight failures
* No user instructions to reconfigure OpenWebUI CORS
* No brittle browser differences

### Typical IIS approach

Use:

* IIS URL Rewrite module
* Application Request Routing (ARR)

High-level concept:

* `https://yourhost/transcriber/` → static app files
* `https://yourhost/openwebui/` → proxied OpenWebUI upstream
* Both share the same origin `https://yourhost`

Then configure the UI’s OpenWebUI base URL as:

* `https://yourhost/openwebui`

#### Notes

Exact IIS ARR rules vary by your OpenWebUI deployment and upstream (Docker, another host, etc.).
If you want, you can standardize on a simple pattern:

* Upstream OpenWebUI: `http://127.0.0.1:3000`
* Public: `https://yourhost/openwebui/`

And create Rewrite rules:

* inbound rule rewriting `/openwebui/{R:1}` → `http://127.0.0.1:3000/{R:1}`
* enable proxy
* preserve host headers as needed

(If you want a ready-to-copy IIS rewrite + ARR recipe, say so; it’s longer and environment-specific.)

---

## CORS guidance and patterns

### When you need CORS

You need CORS configuration when:

* The UI origin differs from OpenWebUI origin

Example:

* UI: `https://transcriber.example.com`
* OpenWebUI: `https://openwebui.example.com`

The browser blocks cross-origin requests unless OpenWebUI allows that origin.

### Best options, in order

1. **Same-origin reverse proxy** (recommended)
2. Configure OpenWebUI `CORS_ALLOW_ORIGIN` to include the UI origin
3. Disable streaming (if streaming fails through certain proxies)

### Streaming considerations

Some reverse proxies buffer or transform SSE streams.
If summary streaming behaves oddly:

* set “Streaming” to `off` in UI
* confirm proxy supports `text/event-stream` without buffering

---

## Security guidance

### Threat model (practical)

* A user enters an OpenWebUI API key into the browser UI.
* If the UI loads third-party scripts, those scripts could exfiltrate the key.
* If the UI has XSS vulnerabilities, attacker could steal the key.
* If the user’s machine is compromised, all bets are off.

### Key handling best practices

* Keep the key in memory only (this app does).
* Do not write to LocalStorage, cookies, or URL params.
* Avoid injecting the key into logs.

### Recommended production hardening

1. **Self-host dependencies**

   * Replace CDN imports with locally hosted assets.
   * This avoids supply-chain and CSP looseness.

2. **Content Security Policy (CSP)**

   * Disallow inline scripts.
   * Allow scripts only from your origin.
   * If you must use a CDN, explicitly allow it.

3. **No third-party analytics**

   * If you require analytics, ensure they cannot access the key.

4. **Set secure headers**

   * `Referrer-Policy: no-referrer`
   * `X-Content-Type-Options: nosniff`
   * `Permissions-Policy` as appropriate

### Permissions Policy

Screen capture is controlled by user gesture and browser permission prompts. You generally do not need to relax Permissions-Policy, but do not restrict it in a way that blocks capture.

---

## Performance guidance

### Baseline recommendations

* ASR model: `whisper-base.en`
* Window: 15s
* Overlap: 3s
* Word timestamps: `word`
* Diarization: off unless needed
* Device: auto (prefer WebGPU)

### WebGPU vs WASM

* WebGPU can make the difference between usable and unusable real-time transcription.
* WASM works but can lag significantly, especially with word timestamps and diarization enabled.

### Diarization cost

Diarization adds:

* VAD segmentation overhead (small)
* embedding inference per segment (can be significant)
* clustering and assignment (small)

If performance is borderline:

* disable diarization
* reduce ASR model size
* increase window size slightly (reduces overhead but increases latency)
* reduce word timestamps (chunk-only)

---

## Troubleshooting

### “No audio track captured”

Symptoms:

* Capture starts, but transcript is empty
* Diagnostics shows “No audio track captured”

Fix:

* In the capture picker, ensure **Share audio** is enabled
* Try capturing a **browser tab** instead of full screen
* Some capture surfaces do not provide system audio

### “Audio is present but transcript is empty”

Possible causes:

* The system audio is silent / low volume
* VAD/ASR windowing not filling
* ASR worker not initialized

Checks:

* In Diagnostics, verify PCM chunks arriving (rms values > ~0.01 for audible audio)
* Confirm ASR model “ready”
* Try “Transcribe now” if auto-run is off

### WebGPU fails

Symptoms:

* Model init errors
* Worker reports WebGPU failure then fallback

Fix:

* Set device preference to `wasm`
* Update browser and GPU drivers
* Test on a different machine to confirm environment constraints

### OpenWebUI “CORS/auth error”

Symptoms:

* Model list fails
* Summaries fail
* DevTools shows CORS error or 401

Fix:

* Verify base URL is correct
* Verify API key is valid
* If it is a CORS error:

  * serve UI and OpenWebUI from same origin, or
  * configure OpenWebUI to allow the UI origin

### Streaming summaries don’t stream

Symptoms:

* Summary appears only at end
* Request hangs
* Proxy buffers

Fix:

* Turn streaming off
* If you control the proxy, disable buffering for `text/event-stream`

### Diarization merges speakers

Fix:

* Lower similarity threshold slightly (e.g., 0.78 → 0.76)
* Ensure VAD segments are long enough (increase `minSegMs` slightly)
* Ensure audio quality is sufficient

### Diarization splits one speaker into many

Fix:

* Raise similarity threshold (e.g., 0.78 → 0.82)
* Increase `minSegMs`
* Increase `hangoverMs` slightly so segments aren’t fragmented

### Diarization doesn’t label anything

Checks:

* Diarization enabled?
* Speaker embedding model loaded?
* Are VAD segments being created?
* Is RMS threshold too high?

Fix:

* Reduce `VAD.rmsThreshold` in code (advanced)
* Ensure word timestamps enabled (speaker-to-word alignment works best)

---

## Exports and integration

### Export formats

#### JSON export (recommended)

Includes:

* session anchors (`captureStartEpochMs`, `firstSampleStart` implicit via conversion)
* transcriptWords: `[{ t0, t1, text, speakerId }]`
* transcriptSegments: window-level segments for debugging
* diarizationSegments: speech segments with speaker IDs
* speakers: cluster IDs + counts

Use JSON export for:

* post-processing
* storage
* re-rendering transcripts in other UIs
* higher-quality deduplication

#### TXT export

* Useful for quick sharing but loses structure (word tokens, precise timing metadata).

### Deduplication (post-processing)

Because window overlap can cause repeated words/phrases:

* Use word timestamps to dedupe:

  * drop words whose `(t0,t1)` overlaps heavily with an existing word
  * or merge by time buckets

---

## Extending the app

High-impact improvements (still local-only):

1. Replace RMS VAD with WebRTC VAD (WASM)

   * Better segmentation
   * Fewer false positives on noise

2. Use a speaker-verification-tuned embedding model

   * Improves clustering reliability

3. Improve clustering

   * Add a “speaker merge” UI
   * Add re-clustering pass on export

4. Better transcript assembly

   * Build a time-ordered word lattice
   * Merge overlaps deterministically by timestamp and confidence

5. Map-reduce summarization

   * For long sessions:

     * segment summaries
     * then merge into final summary
   * Controls token usage and improves stability

6. Offline / self-hosted models

   * Host Transformers.js bundle and model files locally
   * Set strict CSP

---

## FAQ

### Can I host this with IIS?

Yes. This is a static web app and works well on IIS provided:

* HTTPS is enabled
* `.js` MIME type is set correctly (`text/javascript`)
* (optional) `.wasm` MIME type is set (`application/wasm`) if you add WASM assets

See [IIS hosting](#iis-hosting).

### Does the backend ever see audio?

No. Audio stays on the client. Only transcript text is sent to OpenWebUI if you enable summarization.

### Can I make summarization also local?

Yes, but you would need an in-browser LLM or local runtime (e.g., WebGPU LLM) which is significantly heavier.

### Why not use Azure OpenAI directly from the browser?

You explicitly want to avoid per-user Azure keys and avoid CORS issues; OpenWebUI is a better fit for your user base.

### Is diarization “production grade”?

Not equivalent to specialized diarization systems. It’s a practical, static-web-friendly approximation. If you need high accuracy diarization, plan for:

* a dedicated diarization backend, or
* a local native app with better models, or
* a specialized diarization WASM stack (complex)

### Can I embed this in a C# app (WebView2)?

Yes. If you do:

* you can also consider local transcription/summarization in C#
* but you said it’s less scalable; the web app is more scalable for distribution

---

## Operational checklists

### Developer checklist (local)

* [ ] Serve from localhost or HTTPS
* [ ] Test capture with system audio enabled
* [ ] Verify WebGPU availability (optional)
* [ ] Verify ASR worker loads
* [ ] Verify transcript timestamps increment correctly
* [ ] Verify OpenWebUI model list works for a known instance

### Operator checklist (deployment)

* [ ] Host `/app` over HTTPS
* [ ] Confirm MIME types for `.js` (IIS particularly)
* [ ] Confirm `asr-worker.js` loads (no 404)
* [ ] If OpenWebUI is cross-origin, confirm CORS or use reverse proxy
* [ ] If using SSE streaming, confirm proxy doesn’t buffer `text/event-stream`

### Security checklist

* [ ] No third-party scripts unless audited
* [ ] Consider self-hosting Transformers.js bundle
* [ ] Consider CSP restricting scripts to self
* [ ] Do not persist OpenWebUI API key
* [ ] Ensure exports don’t inadvertently include secrets

---

## License



* Apache-2.0

