"""
Tests for SileroVADDetector module.
"""

import numpy as np
from voice_agent2.vad import SileroVADDetector

def test_silero_vad_initialization():
    detector = SileroVADDetector()
    assert detector.sample_rate == 16000
    print("✅ Silero VAD initialization test passed!")

def test_silero_vad_silence_detection():
    detector = SileroVADDetector()
    # Create 512 samples of pure silence (zeros)
    silence_frame = np.zeros(512, dtype=np.float32)
    prob = detector.get_speech_prob(silence_frame)
    assert prob < 0.5
    print(f"✅ Pure silence frame probability test passed! (prob: {prob:.4f})")

def test_silero_vad_synthetic_sine():
    detector = SileroVADDetector()
    # Create 512 samples of sine wave signal
    t = np.linspace(0, 512/16000, 512, endpoint=False)
    sine_frame = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    prob = detector.get_speech_prob(sine_frame)
    assert 0.0 <= prob <= 1.0
    print(f"✅ Synthetic signal probability test passed! (prob: {prob:.4f})")

if __name__ == "__main__":
    test_silero_vad_initialization()
    test_silero_vad_silence_detection()
    test_silero_vad_synthetic_sine()
    print("All Silero VAD unit tests passed successfully!")
