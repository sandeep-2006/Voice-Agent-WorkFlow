"""
Integration test to verify Whisper model loading and audio transcription.
"""

import numpy as np
from voice_agent2.transcriber import WhisperTranscriber

def test_transcription():
    print("Testing WhisperTranscriber with tiny model...")
    transcriber = WhisperTranscriber(model_name="tiny")
    
    # Create 2 seconds of 440Hz sine wave audio at 16kHz sample rate
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    result = transcriber.transcribe_audio_array(sine_wave)
    print(f"Transcription result: '{result.get('text', '')}'")
    print("✅ Transcription engine operational!")

if __name__ == "__main__":
    test_transcription()
