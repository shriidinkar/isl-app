# ISL App (English/Hindi <-> Indian Sign Language)

## Day 1 setup
### Backend
    cd backend
    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    uvicorn app.main:app --reload --port 8000
Open http://localhost:8000/docs

### Frontend
    npm create vite@latest frontend -- --template react
    cd frontend && npm install && npm run dev

### Data
- `data/vocab.csv` = 100 starter words (gloss_id, English, Hindi, category, clip_file, status)
- Put clips in `backend/clips/`, fill `clip_file` + set `status` to `ready`
- `python scripts/check_vocab.py` shows progress

## Sources and licenses (fill in on Day 1)
| Source | License / terms | Link | Permission status |
|---|---|---|---|
