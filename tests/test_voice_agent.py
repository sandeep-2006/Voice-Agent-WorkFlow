"""
Simple unit test for checking voice_agent2 modules import and basic initialization.
"""

import numpy as np

def test_recorder_import():
    from voice_agent2.recorder import check_microphone, SAMPLE_RATE
    assert SAMPLE_RATE == 16000
    print("Recorder module check passed!")

def test_transcriber_import():
    from voice_agent2.transcriber import WhisperTranscriber
    print("Transcriber module check passed!")

if __name__ == "__main__":
    test_recorder_import()
    test_transcriber_import()
    print("All module sanity checks passed!")
