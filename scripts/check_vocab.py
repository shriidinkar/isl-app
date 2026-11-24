"""Run from repo root: python scripts/check_vocab.py
Reports how many vocab entries have a clip file that actually exists."""
import csv
from pathlib import Path

root = Path(__file__).resolve().parents[1]
clips = root / "backend" / "clips"
ok = missing = unassigned = 0
with open(root / "data" / "vocab.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        name = row["clip_file"].strip()
        if not name:
            unassigned += 1
        elif (clips / name).exists():
            ok += 1
        else:
            missing += 1
            print(f"MISSING FILE: {row['gloss_id']} -> {name}")
print(f"\nready: {ok} | file missing: {missing} | no clip yet: {unassigned}")
