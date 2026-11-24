import pytest
from fastapi.testclient import TestClient

from app.gloss import ALIAS_PATH, BY_ID, LOOKUP, text_to_gloss
from app.main import app


def seq(text, **kw):
    return text_to_gloss(text, **kw)["gloss_sequence"]


@pytest.mark.parametrize("text,expected", [
    ("I am going to school", ["I", "SCHOOL", "GO"]),
    ("Hello my name is Priya", ["HELLO", "MY", "NAME", "#PRIYA"]),
    ("What is your name?", ["YOUR", "NAME", "WHAT"]),
    ("I will eat tomorrow", ["TOMORROW", "I", "EAT"]),
    ("I don't want water", ["I", "WATER", "WANT", "NO"]),
    ("Thank you teacher", ["THANK-YOU", "TEACHER"]),
    ("The teachers are learning", ["TEACHER", "LEARN"]),
    ("no", ["NO"]),
    ("मैं स्कूल जाता हूँ", ["I", "SCHOOL", "GO"]),
    ("मैं कल स्कूल जाऊँगा", ["TOMORROW", "I", "SCHOOL", "GO"]),
    ("मैं कल स्कूल गया था", ["YESTERDAY", "I", "SCHOOL", "GO"]),
    ("मैं खाना खाता हूँ", ["I", "FOOD", "EAT"]),
    ("मेरा नाम प्रिया है", ["MY", "NAME", "#PRIYA"]),
    ("क्या तुम स्कूल जाते हो", ["YOU", "SCHOOL", "GO"]),
    ("शुभ रात्रि", ["GOOD-NIGHT"]),
])
def test_sequences(text, expected):
    assert seq(text) == expected


def test_reorder_off_keeps_original_order():
    assert seq("I go to school", reorder_sov=False) == ["I", "GO", "SCHOOL"]


def test_question_flags():
    assert text_to_gloss("क्या तुम स्कूल जाते हो")["is_question"] is True
    assert text_to_gloss("What is your name?")["is_question"] is True
    assert text_to_gloss("I go home")["is_question"] is False


def test_ambiguous_kal_is_flagged():
    r = text_to_gloss("मैं कल स्कूल")
    assert any(i.get("ambiguous") for i in r["items"]) and r["warnings"]


def test_aliases_point_to_real_glosses():
    import csv
    with open(ALIAS_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            assert row["gloss_id"] in BY_ID, row


client = TestClient(app)


def test_api_text_to_isl():
    r = client.post("/text-to-isl", json={"text": "I am going to school"}).json()
    assert r["gloss_sequence"] == ["I", "SCHOOL", "GO"] and r["mode"] == "rules"


def test_api_llm_without_key_falls_back(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client.post("/text-to-isl", json={"text": "I go home", "use_llm": True}).json()
    assert r["mode"] == "rules" and r["warnings"]
