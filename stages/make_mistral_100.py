"""
Build the Mistral annotation over the SAME 100 evaluation reviews as the other
annotators (ChatGPT / Claude / human), so all four can be compared apples-to-apples.

Source = the ORIGINAL Mistral run (`Mistral_predicted_quadruples.csv`), which
contains all 1,000 initial reviews — hence all 100 sample ids. (The *_final* file
is NOT used: it swapped out the 15 sample reviews Mistral parse-failed, so it is
missing them.)

Reviews where Mistral produced no valid quads simply get no rows (an honest
"model found nothing" = recall miss in Stage 3).

Output: data/output/mistral_100_annotation.csv  (schema matches chatgpt_100_annotation.csv)

    python stages/make_mistral_100.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OUTPUT  # noqa: E402

SAMPLE = OUTPUT / "sample_100.csv"
MISTRAL = OUTPUT / "Mistral_predicted_quadruples.csv"   # original run = all 100 ids
OUT = OUTPUT / "mistral_100_annotation.csv"
COLS = ["review_id", "aspect_term", "aspect_category", "opinion_term", "sentiment", "notes"]


def main():
    ids = list(pd.read_csv(SAMPLE)["review_id"].astype(str))
    idset = set(ids)
    m = pd.read_csv(MISTRAL, dtype=str, keep_default_na=False)

    sub = m[m["review_id"].astype(str).isin(idset)].copy()
    # keep only real quad rows (drop the empty parse-fail placeholders)
    sub = sub[sub["aspect_category"].astype(str).str.strip() != ""].copy()
    sub["notes"] = ""
    sub = sub[COLS]
    sub.to_csv(OUT, index=False, encoding="utf-8")

    covered = sorted(idset - set(sub["review_id"].astype(str)))
    print(f"[mistral-100] sample ids: {len(idset)}")
    print(f"[mistral-100] reviews with >=1 Mistral quad: {sub['review_id'].nunique()}")
    print(f"[mistral-100] quads: {len(sub)}")
    print(f"[mistral-100] {len(covered)} sample reviews have NO Mistral quads "
          f"(parse-failed): {covered}")
    print(f"[mistral-100] wrote {OUT}")


if __name__ == "__main__":
    main()
