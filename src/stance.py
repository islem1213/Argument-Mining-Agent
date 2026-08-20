from __future__ import annotations

def estimate_stance(text: str, topic: str | None = None) -> str:
    """Return a transparent pro/con/neutral proxy, not a factuality verdict."""
    lowered = text.lower()
    if any(x in lowered for x in ("not", "never", "harm", "worse", "critics")):
        return "con"
    if any(x in lowered for x in ("should", "improve", "benefit", "support", "better")):
        return "pro"
    return "neutral"
