"""Rule-based text -> ISL gloss engine (English + Hindi).

Pipeline: normalize -> tokenize -> merge phrases -> resolve each token to a gloss
(vocab, aliases, suffix stripping) -> drop function words -> fingerspell unknowns
-> optional ISL-style reordering.

IMPORTANT: the reordering rules (time first, verb late, negation after verb,
question word last) are working hypotheses. Have an ISL signer validate them.
"""
import csv
import re
import unicodedata
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
VOCAB_PATH = BASE_DIR / "data" / "vocab.csv"
ALIAS_PATH = BASE_DIR / "data" / "aliases.csv"


def norm(s: str) -> str:
    """Lowercase + Devanagari normalisation so spelling variants match."""
    s = unicodedata.normalize("NFC", s).lower()
    s = s.replace("\u093c", "")        # nukta: ज़ -> ज
    s = s.replace("\u0901", "\u0902")  # chandrabindu -> anusvara: हाँ -> हां
    s = s.replace("\u200d", "").replace("\u200c", "")
    return s.strip()


def _load():
    with open(VOCAB_PATH, encoding="utf-8", newline="") as f:
        vocab = list(csv.DictReader(f))
    by_id = {v["gloss_id"]: v for v in vocab}
    lookup = {}
    for v in vocab:
        for key in (v["word_en"], v["word_hi"]):
            lookup.setdefault(norm(key), v["gloss_id"])
    if ALIAS_PATH.exists():
        with open(ALIAS_PATH, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                gid = row["gloss_id"].strip()
                if gid in by_id:
                    lookup.setdefault(norm(row["surface"]), gid)
    return vocab, by_id, lookup


VOCAB, BY_ID, LOOKUP = _load()
PHRASES = {k for k in LOOKUP if " " in k}
MAX_PHRASE = max((len(p.split()) for p in PHRASES), default=1)

EN_STOP = set("""a an the is am are was were be been being to of do does did will would shall
can could should must may might in on at for with by from and or but that this these those it its
has have had just also very so as if than then there here""".split())
HI_STOP = {norm(w) for w in """है हैं हूँ हूं हो था थी थे थीं रहा रही रहे का की के को में से ने पर तो ही
भी और कि यह ये इस उस लिए""".split()}
STOP = EN_STOP | HI_STOP
NEG_TOKENS = {"not", "never", norm("न"), norm("ना"), norm("मत")}

KYA = norm("क्या")
KHANA = norm("खाना")
KAL = norm("कल")
HI_PAST = {norm(w) for w in """था थी थे थीं गया गई गयी गए किया लिया खाया आया देखा सोया दिया""".split()}
TIME_FRONT = {"TODAY", "TOMORROW", "YESTERDAY", "MONDAY", "TUESDAY", "WEDNESDAY",
              "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"}
_EAT_FORMS = {k for k, g in LOOKUP.items() if g == "EAT" and k != KHANA}


def tokenize(text: str):
    t = norm(text).replace("\u2019", "'")
    t = re.sub(r"\bcan't\b", "can not", t)
    t = re.sub(r"\bwon't\b", "will not", t)
    t = re.sub(r"n't\b", " not", t)
    t = re.sub(r"'(m|re|s|ve|ll|d)\b", "", t)
    return re.findall(r"[\w\u0900-\u097F]+", t)


def merge_phrases(tokens):
    out, i = [], 0
    while i < len(tokens):
        for n in range(MAX_PHRASE, 1, -1):
            chunk = tokens[i:i + n]
            if len(chunk) == n and " ".join(chunk) in PHRASES:
                out.append(" ".join(chunk))
                i += n
                break
        else:
            out.append(tokens[i])
            i += 1
    return out


def _en_candidates(tok: str):
    if not tok.isascii() or len(tok) < 4:
        return []
    c = []
    if tok.endswith("ing"):
        stem = tok[:-3]
        c += [stem, stem + "e"]
        if len(stem) > 1 and stem[-1] == stem[-2]:
            c.append(stem[:-1])
    if tok.endswith("ed"):
        stem = tok[:-2]
        c += [stem, stem + "e"]
        if len(stem) > 1 and stem[-1] == stem[-2]:
            c.append(stem[:-1])
    if tok.endswith("ies"):
        c.append(tok[:-3] + "y")
    if tok.endswith("es"):
        c.append(tok[:-2])
    if tok.endswith("s"):
        c.append(tok[:-1])
    return [x for x in c if len(x) >= 3]


def _resolve(tok: str):
    """Return gloss id, "" for a dropped function word, or None if unknown."""
    g = LOOKUP.get(tok)
    if g:
        return g
    if tok in STOP:
        return ""
    for cand in _en_candidates(tok):
        if cand in LOOKUP:
            return LOOKUP[cand]
    return None


def _resolve_khana(toks, idx):
    """खाना can mean 'food' or 'to eat'."""
    if any(t in _EAT_FORMS for t in toks):
        return "FOOD", False
    if idx + 1 < len(toks) and toks[idx + 1] == norm("चाहिए"):
        return "FOOD", False
    return "EAT", True


def _resolve_kal(toks, idx):
    """कल can mean 'yesterday' or 'tomorrow'; use tense cues in the sentence."""
    others = toks[:idx] + toks[idx + 1:]
    past = any(t in HI_PAST for t in others)
    future = any(len(t) >= 3 and t.endswith((norm("गा"), norm("गी"), norm("गे"))) for t in others)
    if past and not future:
        return "YESTERDAY", False
    if future and not past:
        return "TOMORROW", False
    return "TOMORROW", True


def _translit(word: str):
    try:
        from indic_transliteration import sanscript
        out = sanscript.transliterate(word, sanscript.DEVANAGARI, sanscript.ITRANS)
    except Exception:
        return None
    out = re.sub(r"[^a-z]", "", out.lower())
    out = re.sub(r"(aa)|(ii)|(uu)", lambda m: m.group(0)[0], out)
    return out or None


def _sign(gloss_id, **extra):
    clip = BY_ID[gloss_id]["clip_file"].strip()
    item = {"type": "sign", "gloss": gloss_id, "clip": f"/clips/{clip}" if clip else None}
    item.update(extra)
    return item


def _fingerspell(word: str):
    if word.isascii():
        return {"type": "fingerspell", "word": word, "letters": list(word)}
    t = _translit(word)
    if t:
        return {"type": "fingerspell", "word": word, "letters": list(t), "approximate": True}
    return {"type": "fingerspell", "word": word, "letters": [], "needs_transliteration": True}


def describe(items):
    return [i["gloss"] if i["type"] == "sign" else "#" + (("".join(i["letters"]) or i["word"]).upper())
            for i in items]


def items_from_sequence(seq):
    """Validate an externally produced sequence (e.g. from an LLM) into items."""
    items = []
    for entry in seq:
        if isinstance(entry, str) and entry in BY_ID:
            items.append(_sign(entry))
        elif isinstance(entry, dict) and isinstance(entry.get("fingerspell"), str) and entry["fingerspell"].strip():
            items.append(_fingerspell(norm(entry["fingerspell"]).replace(" ", "")))
        else:
            raise ValueError(f"invalid gloss entry: {entry!r}")
    return items


def _reorder(entries):
    """entries: list of (item, role). Roles: verb, neg, wh, or None."""
    front = [i for i, r in entries if i.get("gloss") in TIME_FRONT]
    rest = [i for i, r in entries if r is None and i.get("gloss") not in TIME_FRONT]
    verbs = [i for i, r in entries if r == "verb"]
    neg = [i for i, r in entries if r == "neg"]
    wh = [i for i, r in entries if r == "wh"]
    return front + rest + verbs + neg + wh


def text_to_gloss(text: str, reorder_sov: bool = True):
    toks = merge_phrases(tokenize(text))
    is_question = "?" in text
    warnings, entries = [], []

    for idx, tok in enumerate(toks):
        if idx == 0 and tok == KYA and len(toks) > 1:
            is_question = True  # yes/no marker; ISL shows this with facial expression
            continue
        if tok in NEG_TOKENS:
            entries.append((_sign("NO"), "neg"))
            continue

        ambiguous = False
        if tok == KHANA:
            gloss, ambiguous = _resolve_khana(toks, idx)
        elif tok == KAL:
            gloss, ambiguous = _resolve_kal(toks, idx)
        else:
            gloss = _resolve(tok)

        if gloss == "":
            continue
        if gloss is None:
            entries.append((_fingerspell(tok), None))
            continue

        item = _sign(gloss)
        if ambiguous:
            item["ambiguous"] = True
            warnings.append(f"'{tok}' is ambiguous; chose {gloss} from context. Please verify.")
        cat = BY_ID[gloss]["category"]
        entries.append((item, "verb" if cat == "verb" else "wh" if cat == "question" else None))

    if len(entries) > 1:  # a lone "no" stays a plain answer
        entries = [(i, "neg" if i.get("gloss") == "NO" and r is None else r) for i, r in entries]

    items = _reorder(entries) if reorder_sov else [i for i, _ in entries]
    return {"input": text, "items": items, "gloss_sequence": describe(items),
            "is_question": is_question, "mode": "rules", "warnings": warnings}
