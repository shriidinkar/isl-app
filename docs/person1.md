# Person 1: backend / Direction A (Days 1-3)

## Install (from repo root, venv active)
    cd backend
    uv pip install -r requirements-dev.txt
    python -m pytest -q                      # 21 tests should pass
    uvicorn app.main:app --reload --port 8000

First call to /speech-to-text downloads the Whisper model (several hundred MB for "small").
Set WHISPER_MODEL=base for faster/lighter, medium for better Hindi.

## Try it
    curl -X POST localhost:8000/text-to-isl -H 'Content-Type: application/json' \
      -d '{"text":"I am going to school"}'
    curl -X POST localhost:8000/speech-to-text -F file=@sample.m4a -F lang=hi

## Extending the vocabulary
- New sign: add a row to data/vocab.csv.
- New spelling/inflection of an existing sign: add a row to data/aliases.csv (surface,gloss_id).
- Run `python -m pytest -q` after every change.

## Needs signer validation
Word-order rules in gloss.py (_reorder): time first, verb late, negation after verb, question word last.
