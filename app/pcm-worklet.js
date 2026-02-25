// AudioWorkletProcessor: captures audio frames, mixes to mono, downsamples to 16kHz,
// and posts Float32Array chunks with absolute sample indices + RMS for VAD.
//
// Output message shape:
// { pcm: Float32Array, startSample: number, rms: number }
//
// Notes:
// - Downsampling is linear interpolation (fast baseline).
// - RMS is computed on the downsampled chunk.

class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.inRate = sampleRate;
    this.outRate = 16000;
    this.ratio = this.inRate / this.outRate;

    // Absolute sample counter at outRate
    this.outSampleCursor = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || !input[0] || input[0].length === 0) return true;

    const ch0 = input[0];
    const ch1 = input[1];
    const mono = ch1 ? this._mixToMono(ch0, ch1) : ch0;

    const outLen = Math.floor(mono.length / this.ratio);
    if (outLen <= 0) return true;

    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const pos = i * this.ratio;
      const p0 = Math.floor(pos);
      const p1 = Math.min(p0 + 1, mono.length - 1);
      const t = pos - p0;
      out[i] = (1 - t) * mono[p0] + t * mono[p1];
    }

    const startSample = this.outSampleCursor;
    this.outSampleCursor += outLen;

    // RMS for naive VAD / debugging
    let sumSq = 0;
    for (let i = 0; i < outLen; i++) sumSq += out[i] * out[i];
    const rms = Math.sqrt(sumSq / outLen);

    this.port.postMessage({ pcm: out, startSample, rms });
    return true;
  }

  _mixToMono(a, b) {
    const n = Math.min(a.length, b.length);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = 0.5 * (a[i] + b[i]);
    return out;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
