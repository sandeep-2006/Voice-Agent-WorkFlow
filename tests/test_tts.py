"""
Tests for KokoroTTSEngine module.
"""

from voice_agent2.tts import KokoroTTSEngine

def test_tts_initialization():
    engine = KokoroTTSEngine(enabled=True)
    assert engine.enabled is True
    print("✅ KokoroTTSEngine initialization test passed!")

def test_tts_clean_text():
    engine = KokoroTTSEngine(enabled=False)
    cleaned = engine._clean_text_for_speech("# Hello **World**! *Testing* [link]")
    assert cleaned == "Hello World! Testing link"
    print("✅ Text cleaning test passed!")

def test_tts_streaming_buffer():
    engine = KokoroTTSEngine(enabled=False)
    def dummy_generator():
        yield "Hello, "
        yield "this is a test sentence. "
        yield "And another sentence!"
        
    tokens = list(engine.speak_stream(dummy_generator()))
    assert "".join(tokens) == "Hello, this is a test sentence. And another sentence!"
    print("✅ TTS streaming generator test passed!")

if __name__ == "__main__":
    test_tts_initialization()
    test_tts_clean_text()
    test_tts_streaming_buffer()
    print("All TTS unit tests passed successfully!")
