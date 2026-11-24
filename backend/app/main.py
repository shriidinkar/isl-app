import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from . import stt  # noqa: E402
from .gloss import VOCAB, describe, items_from_sequence, text_to_gloss  # noqa: E402
from .llm_gloss import llm_gloss  # noqa: E402

CLIPS_DIR = Path(__file__).resolve().parents[1] / "clips"

app = FastAPI(title="ISL App API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
CLIPS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/clips", StaticFiles(directory=CLIPS_DIR), name="clips")


class TextIn(BaseModel):
    text: str
    lang: str = "en"
    reorder: bool = True    # apply ISL-style word order
    use_llm: bool = False   # optional; needs ANTHROPIC_API_KEY


@app.get("/health")
def health():
    return {"status": "ok", "vocab_size": len(VOCAB)}


@app.get("/vocab")
def vocab():
    return VOCAB


@app.post("/text-to-isl")
def text_to_isl(body: TextIn):
    result = text_to_gloss(body.text, reorder_sov=body.reorder)
    if body.use_llm:
        try:
            items = items_from_sequence(llm_gloss(body.text))
            result.update(items=items, gloss_sequence=describe(items), mode="llm")
        except Exception as e:  # any failure -> keep the rule-based result
            result["warnings"].append(f"LLM mode unavailable, used rules ({type(e).__name__}: {str(e)[:100]})")
    return result


@app.post("/speech-to-text")
def speech_to_text(file: UploadFile = File(...), lang: str | None = Form(None)):
    suffix = Path(file.filename or "audio").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name
    try:
        return stt.transcribe_file(path, lang)
    except ImportError:
        raise HTTPException(500, "faster-whisper is not installed. Run: uv pip install -r requirements.txt")
    finally:
        os.unlink(path)
