// frontend/public/audio-worklet.js

class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.prevInput = 0;
    this.prevOutput = 0;
  }

  process(inputs) {
    const input = inputs[0][0];
    if (!input) return true;

    const output = new Float32Array(input.length);

    const GAIN = 1.6;        // 🔊 SAFE gain (do not exceed 2.0)
    const NOISE_GATE = 0.002; // 🔇 Noise floor
    const HPF = 0.995;       // 🎚️ High-pass filter constant

    for (let i = 0; i < input.length; i++) {
      let x = input[i];

      // ✅ Proper DC-offset removal (1st order HPF)
      let y = x - this.prevInput + HPF * this.prevOutput;
      this.prevInput = x;
      this.prevOutput = y;

      // 🔇 Noise gate
      if (Math.abs(y) < NOISE_GATE) y = 0;

      // 🔊 Apply safe gain
      y *= GAIN;

      // 🛑 Hard clip protection
      if (y > 1) y = 1;
      if (y < -1) y = -1;

      output[i] = y;
    }

    // Send clean float32 PCM to main thread
    this.port.postMessage(output);

    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);
