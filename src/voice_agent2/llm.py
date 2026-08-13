"""
LLM Integration Module for Voice-to-Text Pipeline with Real-time Streaming Output.
Supports OpenAI, Google Gemini, Groq, Ollama, and Fallback mode.
"""

import os
import datetime
from typing import List, Dict, Generator
from dotenv import load_dotenv

load_dotenv()

DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable, friendly, and intelligent AI voice assistant. "
    "Provide short length, well-structured, and comprehensive responses that cover all key details and information requested. "
    "Ensure your tone is natural, clear, and engaging when read aloud, keeping date and context accurate."
)

class LLMEngine:
    def __init__(
        self,
        provider: str = None,
        model_name: str = None,
        system_prompt: str = None,
    ):
        today_date = datetime.datetime.now().strftime("%B %d, %Y")
        base_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.system_prompt = f"{base_prompt}\nToday's date is {today_date}."
        self.provider = provider or os.getenv("LLM_PROVIDER", "auto").lower()
        self.model_name = model_name
        self.history: List[Dict[str, str]] = []

        if self.provider == "auto":
            self.provider = self._auto_detect_provider()

        if not self.model_name:
            self.model_name = self._default_model_for_provider(self.provider)

        print(f"🤖 Initialized LLM Engine [Provider: {self.provider.upper()}, Model: {self.model_name}]")

    def _auto_detect_provider(self) -> str:
        if os.getenv("GROQ_API_KEY"):
            return "groq"
        elif os.getenv("OPENAI_API_KEY"):
            return "openai"
        elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            return "gemini"
        else:
            return "groq"

    def _default_model_for_provider(self, provider: str) -> str:
        defaults = {
            "openai": "gpt-4o-mini",
            "gemini": "gemini-2.5-flash",
            "groq": "llama-3.3-70b-versatile",
            "ollama": "llama3.2",
            "fallback": "echo-assistant",
        }
        return defaults.get(provider, "llama-3.3-70b-versatile")

    def reset_conversation(self):
        self.history = []
        print("🧹 Conversation history reset.")

    def generate_response_stream(self, user_text: str) -> Generator[str, None, None]:
        """
        Generate streaming token response from LLM for the given user prompt.
        Yields text chunks as they arrive from the API.
        """
        if not user_text or not user_text.strip():
            yield "I didn't catch any text. Please try speaking again."
            return

        self.history.append({"role": "user", "content": user_text})
        full_response = ""

        try:
            if self.provider == "groq":
                generator = self._stream_groq(self.history)
            elif self.provider == "openai":
                generator = self._stream_openai(self.history)
            elif self.provider == "gemini":
                generator = self._stream_gemini(self.history)
            elif self.provider == "ollama":
                generator = self._stream_ollama(self.history)
            else:
                generator = self._stream_fallback(user_text)

            for chunk in generator:
                full_response += chunk
                yield chunk

        except Exception as e:
            err_msg = f"[LLM Error ({self.provider})]: {e}\n\nPlease check your API key or model configuration."
            full_response += err_msg
            yield err_msg

        self.history.append({"role": "assistant", "content": full_response})

    def generate_response(self, user_text: str) -> str:
        """
        Non-streaming helper method for convenience.
        """
        return "".join(self.generate_response_stream(user_text))

    def _stream_groq(self, history: List[Dict[str, str]]) -> Generator[str, None, None]:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            yield (
                "⚠️ GROQ_API_KEY is not set!\n"
                "Please set GROQ_API_KEY=gsk_... in your environment or .env file.\n"
                "You can get a free key at https://console.groq.com"
            )
            return

        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        messages = [{"role": "system", "content": self.system_prompt}] + history
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta

    def _stream_openai(self, history: List[Dict[str, str]]) -> Generator[str, None, None]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            yield "⚠️ OPENAI_API_KEY is not set in environment or .env file!"
            return

        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        messages = [{"role": "system", "content": self.system_prompt}] + history
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta

    def _stream_gemini(self, history: List[Dict[str, str]]) -> Generator[str, None, None]:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            yield "⚠️ GEMINI_API_KEY is not set in environment or .env file!"
            return

        from google import genai
        client = genai.Client(api_key=api_key)
        contents = [self.system_prompt + "\n\nConversation history:"]
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            contents.append(f"{role}: {msg['content']}")
        contents.append("Assistant:")
        prompt = "\n".join(contents)

        response = client.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
        )
        for chunk in response:
            text_chunk = chunk.text or ""
            if text_chunk:
                yield text_chunk

    def _stream_ollama(self, history: List[Dict[str, str]]) -> Generator[str, None, None]:
        from openai import OpenAI
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434/v1")
        client = OpenAI(
            api_key="ollama",
            base_url=host,
        )
        messages = [{"role": "system", "content": self.system_prompt}] + history
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta

    def _stream_fallback(self, user_text: str) -> Generator[str, None, None]:
        msg = (
            f"Received: \"{user_text}\"\n"
            f"💡 Note: No LLM API key detected. Set GROQ_API_KEY, OPENAI_API_KEY, "
            f"or GEMINI_API_KEY in your environment or .env file to enable live LLM responses!"
        )
        yield msg
