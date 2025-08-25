# ai/providers.py — Multi-provider abstraction (per-request keys, auto temperature)
from typing import Optional, Dict, List
import re
import requests
import google.generativeai as genai
from google.api_core import client_options as client_options_lib
from openai import OpenAI
try:
    import anthropic
    _ANTHROPIC = True
except Exception:
    _ANTHROPIC = False

JSON_HINT = "Return ONLY valid JSON."

class AIResponseError(Exception):
    pass

def _strip_fences(s: str) -> str:
    if not s:
        return ""
    # Remove common markdown fences and surrounding whitespace
    s = s.strip()
    s = re.sub(r"^```(?:json|txt)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

class ProviderBase:
    name: str
    def __init__(self, temperature: float = 0.2): self.temperature = temperature
    def generate_text(self, prompt: str, json_mode: bool, keys: Dict[str, Optional[str]]) -> str:
        raise NotImplementedError

# -------- Gemini --------
class GeminiProvider(ProviderBase):
    name = "gemini"
    def __init__(self, model: str = "gemini-1.5-flash", temperature: float = 0.2):
        super().__init__(temperature); self.model_name = model
    def generate_text(self, prompt: str, json_mode: bool, keys: Dict[str, Optional[str]]) -> str:
        api_key = (keys or {}).get("gemini")
        if not api_key: raise AIResponseError("Gemini key missing.")
        genai.configure(
            api_key=api_key,
            transport="rest",
            client_options=client_options_lib.ClientOptions(api_endpoint="generativelanguage.googleapis.com"),
        )
        model = genai.GenerativeModel(self.model_name)
        p = f"{JSON_HINT}\n{prompt}" if json_mode else prompt
        out = model.generate_content(
            p,
            generation_config={"temperature": self.temperature},
            request_options={"timeout": 60}
        )
        return _strip_fences(out.text or "")

# -------- OpenAI --------
class OpenAIProvider(ProviderBase):
    name = "openai"
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        super().__init__(temperature); self.model_name = model
    def generate_text(self, prompt: str, json_mode: bool, keys: Dict[str, Optional[str]]) -> str:
        api_key = (keys or {}).get("openai")
        if not api_key: raise AIResponseError("OpenAI key missing.")
        client = OpenAI(api_key=api_key, timeout=60.0)
        messages = [{"role": "user", "content": prompt}]
        if json_mode:
            messages.insert(0, {"role": "system", "content": JSON_HINT})
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"} if json_mode else {"type": "text"},
        )
        return _strip_fences(resp.choices[0].message.content or "")

# -------- Anthropic --------
class ClaudeProvider(ProviderBase):
    name = "anthropic"
    def __init__(self, model: str = "claude-3-haiku-20240307", temperature: float = 0.2):
        super().__init__(temperature); self.model_name = model
        if not _ANTHROPIC: raise AIResponseError("Anthropic lib not installed.")
    def generate_text(self, prompt: str, json_mode: bool, keys: Dict[str, Optional[str]]) -> str:
        api_key = (keys or {}).get("anthropic")
        if not api_key: raise AIResponseError("Anthropic key missing.")
        client = anthropic.Anthropic(api_key=api_key, timeout=60.0)
        system = JSON_HINT if json_mode else ""
        msg = client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        text = "".join(b.text for b in msg.content if hasattr(b, "text"))
        return _strip_fences(text)

# -------- Ollama (local) --------
class OllamaProvider(ProviderBase):
    name = "ollama"
    def __init__(self, temperature: float = 0.2): super().__init__(temperature)
    def generate_text(self, prompt: str, json_mode: bool, keys: Dict[str, Optional[str]]) -> str:
        base = (keys or {}).get("ollama_base") or "http://localhost:11434"
        model = (keys or {}).get("ollama_model") or "llama3.1"
        messages = [{"role": "user", "content": prompt}]
        if json_mode: messages.insert(0, {"role": "system", "content": JSON_HINT})
        r = requests.post(
            f"{base}/api/chat",
            json={"model": model, "messages": messages, "stream": False, "format": "json" if json_mode else "text"},
            timeout=120
        )
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "")
        return _strip_fences(content)

# -------- Manager with fallback --------
class ProviderManager:
    def __init__(self, order: Optional[List[str]] = None, temperature: float = 0.2):
        self.order = order or ["gemini","openai","anthropic","ollama"]
        self.temperature = temperature
    def _instantiate(self, name: str):
        if name == "gemini":     return GeminiProvider(temperature=self.temperature)
        if name == "openai":     return OpenAIProvider(temperature=self.temperature)
        if name == "anthropic":  return ClaudeProvider(temperature=self.temperature)
        if name == "ollama":     return OllamaProvider(temperature=self.temperature)
        raise AIResponseError(f"Unknown provider {name}")
    def generate(self, prompt: str, json_mode: bool, keys: Dict[str, Optional[str]], order: Optional[List[str]] = None) -> str:
        last_err = None
        prompt = prompt.get("raw_text", "") if isinstance(prompt, dict) else str(prompt or "")
        for name in (order or self.order):
            try:
                prov = self._instantiate(name)
                out = prov.generate_text(prompt, json_mode=json_mode, keys=keys)
                return out.strip()
            except Exception as e:
                last_err = e
                continue
        raise AIResponseError(f"All providers failed. Last error: {last_err}")
