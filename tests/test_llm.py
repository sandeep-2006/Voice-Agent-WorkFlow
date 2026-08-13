"""
Tests for LLMEngine streaming and provider integration.
"""

import os
from voice_agent2.llm import LLMEngine

def test_groq_streaming_provider():
    engine = LLMEngine(provider="groq")
    assert engine.provider == "groq"
    assert engine.model_name == "llama-3.3-70b-versatile"
    
    # Test streaming response generator
    chunks = list(engine.generate_response_stream("Hello Groq"))
    full_text = "".join(chunks)
    assert len(full_text) > 0
    if not os.getenv("GROQ_API_KEY"):
        assert "GROQ_API_KEY is not set" in full_text
    print("✅ Groq streaming test passed!")

def test_llm_reset():
    engine = LLMEngine(provider="groq")
    engine.generate_response("Test message")
    assert len(engine.history) == 2
    engine.reset_conversation()
    assert len(engine.history) == 0
    print("✅ LLMEngine reset test passed!")

if __name__ == "__main__":
    test_groq_streaming_provider()
    test_llm_reset()
    print("All LLM streaming tests passed successfully!")
