"""Optional LLM gloss mode. Needs ANTHROPIC_API_KEY (Claude Console API credits,
billed separately from a Claude.ai subscription). Falls back to rules on any error."""
import json
import os
import re

from .gloss import VOCAB

SYSTEM = """You convert English or Hindi sentences into Indian Sign Language (ISL) gloss sequences for a small fixed vocabulary.
Rules:
- Use ONLY gloss IDs from the vocabulary list in the user message.
- Drop function words (articles, auxiliaries, prepositions, tense markers).
- Put time words (TODAY, TOMORROW, YESTERDAY, day names) first.
- Put the verb near the end, then negation (NO), then question words last.
- For names or words not in the vocabulary, output {"fingerspell": "<word in Latin letters>"}.
- Output ONLY a JSON array. No other text."""


class LLMUnavailable(Exception):
    pass


def llm_gloss(text: str):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise LLMUnavailable("ANTHROPIC_API_KEY not set")
    import anthropic  # imported lazily so the app runs without it

    vocab = ", ".join(f"{v['gloss_id']}={v['word_en']}/{v['word_hi']}" for v in VOCAB)
    client = anthropic.Anthropic(api_key=key)
    resp = client.messages.create(
        model=os.getenv("GLOSS_MODEL", "claude-haiku-5-5"),
        max_tokens=400,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Vocabulary:\n{vocab}\n\nSentence: {text}"}],
    )
    raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    seq = json.loads(raw)
    if not isinstance(seq, list):
        raise ValueError("LLM did not return a JSON array")
    return seq
