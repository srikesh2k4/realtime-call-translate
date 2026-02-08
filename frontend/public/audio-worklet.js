// frontend/public/audio-worklet.js

class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // DC-offset removal (high-pass filter state)
    this.prevInput = 0;
    this.prevOutput = 0;

    // ===== Adaptive Noise Gate =====
    // We track the noise floor dynamically and only let through
    // audio that is clearly above it (i.e. voice).
    this.noiseFloor = 0.005;       // initial noise estimate
    this.NOISE_ADAPT_UP = 0.0002;  // how fast noise floor rises (slow)
    this.NOISE_ADAPT_DOWN = 0.001; // how fast it drops when silence detected
    this.GATE_RATIO = 3.0;         // signal must be 3× above noise floor

    // ===== Smooth Gain Envelope =====
    // Prevents harsh on/off clicks by ramping gain up/down.
    this.envelope = 0;              // current gain envelope (0..1)
    this.ATTACK = 0.05;             // ramp-up speed (per sample)
    this.RELEASE = 0.005;           // ramp-down speed (per sample)

    // ===== Frame-level Voice Activity Detection =====
    // We compute RMS and zero-crossing rate per frame.
    // Voice has moderate ZCR and higher RMS.
    this.silenceFrames = 0;         // consecutive silent frames
    this.MAX_SILENCE_FRAMES = 12;   // ~96ms at 128-sample frames @ 16kHz
    this.isVoiceActive = false;

    // ===== Output gain =====
    this.GAIN = 1.4;                // output amplification
  }

  process(inputs) {
    const input = inputs[0][0];
    if (!input) return true;

    const output = new Float32Array(input.length);
    const HPF = 0.995; // high-pass filter constant

    // --- Frame-level stats ---
    let frameRMS = 0;
    let zeroCrossings = 0;
    let prevSample = 0;

    // First pass: compute frame statistics
    for (let i = 0; i < input.length; i++) {
      let x = input[i];

      // DC offset removal (1st order HPF)
      let y = x - this.prevInput + HPF * this.prevOutput;
      this.prevInput = x;
      this.prevOutput = y;

      frameRMS += y * y;

      // Count zero crossings (voice typically 1000-3000 Hz → moderate ZCR)
      if (i > 0 && ((y > 0 && prevSample < 0) || (y < 0 && prevSample > 0))) {
        zeroCrossings++;
      }
      prevSample = y;

      output[i] = y; // store HPF'd samples for second pass
    }

    frameRMS = Math.sqrt(frameRMS / input.length);
    const zcRate = zeroCrossings / input.length; // normalized 0..1

    // --- Adaptive noise floor update ---
    if (frameRMS < this.noiseFloor * 1.5) {
      // Silence/noise region → slowly adapt noise floor
      this.noiseFloor =
        this.noiseFloor * (1 - this.NOISE_ADAPT_DOWN) +
        frameRMS * this.NOISE_ADAPT_DOWN;
    } else {
      // Signal present → slowly raise noise floor (room conditions change)
      this.noiseFloor += this.NOISE_ADAPT_UP;
    }

    // Clamp noise floor to sane range
    this.noiseFloor = Math.max(0.001, Math.min(0.05, this.noiseFloor));

    // --- Voice Activity Decision ---
    // Voice criteria:
    //   1. RMS well above noise floor
    //   2. ZCR not too high (pure noise has very high ZCR > 0.4)
    //   3. ZCR not too low (DC or ultra-low-freq rumble < 0.01)
    const aboveNoise = frameRMS > this.noiseFloor * this.GATE_RATIO;
    const goodZCR = zcRate > 0.02 && zcRate < 0.35;
    const voiceDetected = aboveNoise && goodZCR;

    if (voiceDetected) {
      this.silenceFrames = 0;
      this.isVoiceActive = true;
    } else {
      this.silenceFrames++;
      if (this.silenceFrames > this.MAX_SILENCE_FRAMES) {
        this.isVoiceActive = false;
      }
    }

    // --- Second pass: apply smooth envelope gating + gain ---
    // Reset HPF state for re-filter (we stored HPF'd values in output[])
    for (let i = 0; i < output.length; i++) {
      let y = output[i];

      // Smooth envelope
      if (this.isVoiceActive) {
        this.envelope = Math.min(1.0, this.envelope + this.ATTACK);
      } else {
        this.envelope = Math.max(0.0, this.envelope - this.RELEASE);
      }

      // Apply envelope gating
      y *= this.envelope;

      // Apply gain
      y *= this.GAIN;

      // Hard clip protection
      if (y > 1) y = 1;
      if (y < -1) y = -1;

      output[i] = y;
    }

    // Only send if there's voice activity (saves bandwidth & reduces noise)
    if (this.isVoiceActive || this.envelope > 0.01) {
      this.port.postMessage(output);
    }

    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);
