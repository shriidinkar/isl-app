"""Speech-to-text with faster-whisper (CPU, int8). Model downloads on first use."""
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_model():
    from faster_whisper import WhisperModel  # lazy: app starts even if not installed
    return WhisperModel(os.getenv("WHISPER_MODEL", "small"), device="cpu", compute_type="int8")


def transcribe_file(path: str, lang_hint: str | None = None):
    model = get_model()
    hint = lang_hint if lang_hint in ("en", "hi") else None
    segments, info = model.transcribe(path, language=hint, vad_filter=True, beam_size=5)
    text = " ".join(s.text.strip() for s in segments).strip()
    lang = info.language
    if hint is None and lang not in ("en", "hi"):
        # Spoken Hindi is often detected as Urdu etc.; this app supports en/hi only.
        segments, info = model.transcribe(path, language="hi", vad_filter=True, beam_size=5)
        text = " ".join(s.text.strip() for s in segments).strip()
        lang = "hi"
    return {"text": text, "lang": lang, "duration": round(info.duration, 1)}
