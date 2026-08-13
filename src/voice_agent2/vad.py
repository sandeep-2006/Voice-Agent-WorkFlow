"""
Silero VAD (Voice Activity Detection) Module for Hands-Free Speech Detection & Interruption.
"""

import sys
import time
import torch
import numpy as np
import sounddevice as sd
from silero_vad import load_silero_vad

SAMPLE_RATE = 16000
FRAME_SIZE = 512  # 32ms frames at 16kHz

class SileroVADDetector:
    def __init__(self, threshold: float = 0.5, sample_rate: int = SAMPLE_RATE):
        self.threshold = threshold
        self.sample_rate = sample_rate
        print("⚡ Initializing Silero VAD (Voice Activity Detector)...")
        self.model = load_silero_vad()
        print("✅ Silero VAD initialized successfully!")

    def get_speech_prob(self, frame: np.ndarray) -> float:
        """
        Calculate speech probability (0.0 to 1.0) for a 512-sample float32 array.
        """
        if len(frame) != FRAME_SIZE:
            # Pad or trim to exactly FRAME_SIZE
            if len(frame) < FRAME_SIZE:
                frame = np.pad(frame, (0, FRAME_SIZE - len(frame)))
            else:
                frame = frame[:FRAME_SIZE]
        
        tensor = torch.from_numpy(frame.astype(np.float32))
        with torch.no_grad():
            prob = self.model(tensor, self.sample_rate).item()
        return prob

    def record_audio_vad(
        self,
        silence_duration: float = 1.0,
        min_speech_duration: float = 0.3,
        tts_engine=None,
    ) -> np.ndarray:
        """
        Hands-free continuous microphone stream.
        - Automatically detects when user starts speaking.
        - Interrupts TTS if active.
        - Automatically stops when user pauses for silence_duration (default 1.0s).
        Returns mono float32 audio array for Whisper.
        """
        print("\n🎧 Silero VAD Listening... Speak anytime! (Auto-detects speech & silence)")
        
        recorded_chunks = []
        is_speaking = False
        silence_start_time = None
        speech_start_time = None

        # Reset model state before new recording
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()

        def callback(indata, frames, time_info, status):
            nonlocal is_speaking, silence_start_time, speech_start_time
            if status:
                print(f"Audio status warning: {status}", file=sys.stderr)

            audio_chunk = indata.flatten()
            prob = self.get_speech_prob(audio_chunk)

            if prob >= self.threshold:
                # Speech detected!
                if not is_speaking:
                    is_speaking = True
                    speech_start_time = time.time()
                    print("\n🔴 Speech Detected! Recording voice...", flush=True)

                    # If TTS is speaking, interrupt it immediately!
                    if tts_engine is not None:
                        tts_engine.interrupt()

                silence_start_time = None
                recorded_chunks.append(audio_chunk.copy())
            else:
                # Silence frame
                if is_speaking:
                    recorded_chunks.append(audio_chunk.copy())
                    if silence_start_time is None:
                        silence_start_time = time.time()

        # Open stream with 512 frame blocksize
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SIZE,
            callback=callback,
        )

        with stream:
            while True:
                time.sleep(0.05)
                # Check if user has finished speaking (silence after speech)
                if is_speaking and silence_start_time is not None:
                    silence_elapsed = time.time() - silence_start_time
                    if silence_elapsed >= silence_duration:
                        speech_elapsed = time.time() - speech_start_time
                        if speech_elapsed >= min_speech_duration:
                            print(f"\n⏹️  Silence detected ({silence_elapsed:.1f}s). End of speech.")
                            break

        if not recorded_chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate(recorded_chunks, axis=0).flatten()
