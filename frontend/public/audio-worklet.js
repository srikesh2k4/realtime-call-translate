// frontend/public/audio-worklet.js
// Optimized PCM Processor with advanced noise suppression

class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    
    // High-pass filter state (removes DC offset and low rumble)
    this.prevInput = 0;
    this.prevOutput = 0;
    
    // Noise gate state
    this.gateOpen = false;
    this.gateHoldSamples = 0;
    
    // Smoothing for noise gate (prevents choppy audio)
    this.envelope = 0;
    
    // Circular buffer for noise floor estimation
    this.noiseFloorBuffer = new Float32Array(50);
    this.noiseFloorIndex = 0;
    this.noiseFloorEstimate = 0.002;
    
    // Voice activity detection
    this.vadHistory = new Float32Array(10);
    this.vadIndex = 0;
  }

  // Calculate RMS energy of a buffer
  calculateRMS(buffer) {
    let sum = 0;
    for (let i = 0; i < buffer.length; i++) {
      sum += buffer[i] * buffer[i];
    }
    return Math.sqrt(sum / buffer.length);
  }

  // Update noise floor estimate during silence
  updateNoiseFloor(rms) {
    // Only update during likely silence (low energy)
    if (rms < this.noiseFloorEstimate * 2) {
      this.noiseFloorBuffer[this.noiseFloorIndex] = rms;
      this.noiseFloorIndex = (this.noiseFloorIndex + 1) % this.noiseFloorBuffer.length;
      
      // Median filter for robust estimation
      const sorted = [...this.noiseFloorBuffer].sort((a, b) => a - b);
      this.noiseFloorEstimate = sorted[Math.floor(sorted.length / 2)];
    }
  }

  // Voice activity detection
  isVoiceActive(rms) {
    // Threshold is adaptive based on noise floor
    const threshold = Math.max(0.003, this.noiseFloorEstimate * 3);
    const active = rms > threshold;
    
    // Update VAD history
    this.vadHistory[this.vadIndex] = active ? 1 : 0;
    this.vadIndex = (this.vadIndex + 1) % this.vadHistory.length;
    
    // Voice is active if majority of recent frames are active
    const activeCount = this.vadHistory.reduce((a, b) => a + b, 0);
    return activeCount > this.vadHistory.length * 0.3;
  }

  process(inputs) {
    const input = inputs[0][0];
    if (!input) return true;

    const output = new Float32Array(input.length);
    const rms = this.calculateRMS(input);
    
    // Update noise floor during silence
    this.updateNoiseFloor(rms);
    
    // Check voice activity
    const voiceActive = this.isVoiceActive(rms);

    // Configuration
    const GAIN = 1.4;              // Safe gain
    const HPF = 0.995;             // High-pass filter constant
    const ATTACK = 0.01;           // Fast attack
    const RELEASE = 0.0005;        // Slow release for smooth trailing
    const GATE_HOLD_MS = 150;      // Hold gate open for 150ms after voice
    const GATE_HOLD_SAMPLES = (GATE_HOLD_MS / 1000) * sampleRate;

    // Noise gate logic
    if (voiceActive) {
      this.gateOpen = true;
      this.gateHoldSamples = GATE_HOLD_SAMPLES;
    } else if (this.gateHoldSamples > 0) {
      this.gateHoldSamples -= input.length;
    } else {
      this.gateOpen = false;
    }

    for (let i = 0; i < input.length; i++) {
      let x = input[i];

      // High-pass filter for DC removal and low frequency rejection
      let y = x - this.prevInput + HPF * this.prevOutput;
      this.prevInput = x;
      this.prevOutput = y;

      // Envelope follower for smooth gating
      const targetEnvelope = this.gateOpen ? 1 : 0;
      const rate = targetEnvelope > this.envelope ? ATTACK : RELEASE;
      this.envelope += rate * (targetEnvelope - this.envelope);

      // Apply envelope (smooth noise gate)
      y *= this.envelope;

      // Apply gain
      y *= GAIN;

      // Soft clipping (smoother than hard clip)
      if (y > 0.95) {
        y = 0.95 + (y - 0.95) * 0.1;
      } else if (y < -0.95) {
        y = -0.95 + (y + 0.95) * 0.1;
      }
      
      // Final hard clip protection
      y = Math.max(-1, Math.min(1, y));

      output[i] = y;
    }

    // Only send if gate is open (voice detected)
    if (this.gateOpen || this.envelope > 0.01) {
      this.port.postMessage(output);
    }

    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);
