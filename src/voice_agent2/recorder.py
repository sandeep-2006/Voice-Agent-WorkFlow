"""
Audio recorder module using sounddevice to capture microphone input.
"""

import sys
import threading
import time
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000  # Whisper native sample rate (16kHz mono)

def check_microphone() -> bool:
    """Verify that a default input device (microphone) is available."""
    try:
        devices = sd.query_devices()
        input_devices = [d for d in devices if d.get('max_input_channels', 0) > 0]
        return len(input_devices) > 0
    except Exception as e:
        print(f"Error checking audio devices: {e}")
        return False

def record_audio_push_to_talk(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Record audio until the user presses Enter.
    Returns mono float32 numpy array.
    """
    print("\n🎤 Press [ENTER] to START recording...")
    input()

    audio_chunks = []
    stop_event = threading.Event()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Audio status warning: {status}", file=sys.stderr)
        audio_chunks.append(indata.copy())

    print("🔴 Recording... Press [ENTER] again to STOP recording and transcribe.")
    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', callback=callback)
    
    with stream:
        input()
        stop_event.set()

    print("⏹️  Recording stopped. Processing audio...")

    if not audio_chunks:
        return np.array([], dtype=np.float32)

    audio_data = np.concatenate(audio_chunks, axis=0).flatten()
    return audio_data

def record_audio_fixed(duration: float = 5.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Record audio for a fixed duration in seconds.
    Returns mono float32 numpy array.
    """
    print(f"\n🔴 Recording for {duration} seconds... Speak now!")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    
    start_time = time.time()
    while time.time() - start_time < duration:
        remaining = duration - (time.time() - start_time)
        print(f"\r⏱️  Time remaining: {remaining:.1f}s", end="", flush=True)
        time.sleep(0.1)
    
    sd.wait()
    print("\n⏹️  Recording finished. Processing audio...")

    return audio_data.flatten()
