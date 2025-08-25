# ai/selector.py — Decide provider order dynamically based on keys & strategy
from typing import Dict, List

def choose_order(keys: Dict[str, str], strategy: str = "Best quality") -> List[str]:
    # Guard against None and whitespace-only keys
    keys = keys or {}
    have = {k for k, v in keys.items() if (v or "").strip() and k in {"openai", "gemini", "anthropic", "ollama_base"}}
    
    # Strategy presets
    if strategy == "Best quality":
        pref = ["openai", "anthropic", "gemini", "ollama"]
    elif strategy == "Fastest":
        pref = ["gemini", "openai", "anthropic", "ollama"]
    else:  # Local first
        pref = ["ollama", "gemini", "openai", "anthropic"]
        
    out: List[str] = []
    for p in pref:
        if p == "ollama":
            if "ollama_base" in have:
                out.append("ollama")
        else:
            if p in have:
                out.append(p)
                
    # Fallback if nothing provided, avoid crash
    if not out:
        out = ["gemini", "openai", "anthropic", "ollama"]
        
    return out
