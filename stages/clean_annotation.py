"""
Clean an LLM annotation of the 100-review evaluation sample into a tidy CSV.

Reusable for each annotation source (ChatGPT / Claude / Mistral / human export).
Defaults to the ChatGPT (GPT-5.5) annotation:
    Input : data/output/chatgpt_100_raw.csv        (raw sheet export)
    Output: data/output/chatgpt_100_annotation.csv (one row per quadruple)

Fixes applied:
- exact-duplicate quad rows (the doubled blocks on the long reviews) -> keep one
- trailing / fully-empty rows -> dropped
- malformed / column-shifted rows (aspect_category not one of the 7) -> dropped + reported
- ragged trailing columns -> ignored (only the 7 schema columns are kept)
- cp1252 decode + typographic punctuation normalized to ASCII; mojibake repaired
- sentiment normalized to lowercase; invalid sentiments reported
- reviews annotated as 'no aspects' -> listed (they contribute no quads)

    python stages/clean_annotation.py
    python stages/clean_annotation.py --src data/output/claude_100_raw.csv \\
                                      --out data/output/claude_100_annotation.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ASPECT_CATEGORIES, OUTPUT  # noqa: E402

VALID_CATS = set(ASPECT_CATEGORIES.keys())
VALID_SENT = {"positive", "negative", "neutral"}
COLS = ["review_id", "aspect_term", "aspect_category", "opinion_term", "sentiment", "notes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(OUTPUT / "chatgpt_100_raw.csv"))
    ap.add_argument("--out", default=str(OUTPUT / "chatgpt_100_annotation.csv"))
    args = ap.parse_args()
    RAW, OUT = Path(args.src), Path(args.out)

    if not RAW.exists():
        sys.exit(f"[clean] missing input: {RAW}\n"
                 f"        Export the annotation sheet to that path first.")

    import io

    # Sheets/Excel on Windows saves cp1252. Decode bytes as cp1252 (maps smart quotes
    # correctly), replacing the few undefined bytes, then parse from memory.
    raw_bytes = RAW.read_bytes()
    text = raw_bytes.decode("cp1252", errors="replace")
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False,
                     engine="python", on_bad_lines="skip")
    print(f"[clean] decoded cp1252; {len(text)} chars")

    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in COLS:
        if c not in df.columns:
            df[c] = ""

    # normalize typographic punctuation to ASCII so annotation spans match cleanly
    punct = {"‘": "'", "’": "'", "“": '"', "”": '"',
             "–": "-", "—": "-", "…": "...", " ": " "}
    def norm(s: str) -> str:
        # repair pre-existing double-encoded UTF-8 mojibake (e.g. the "European" cell)
        if "â" in s or "Ã" in s or "�" in s:
            s = s.encode("cp1252", "ignore").decode("utf-8", "ignore")
        for k, v in punct.items():
            s = s.replace(k, v)
        return s.replace("�", "").strip().strip('"').strip()
    for c in COLS:
        df[c] = df[c].astype(str).map(norm)
    n_raw = len(df)

    # 1) drop fully-empty / trailing rows
    empty = (df["review_id"] == "") & (df["aspect_category"] == "")
    n_empty = int(empty.sum())
    df = df[~empty].copy()

    # 2) reviews explicitly marked 'no aspects' (blank category) -> record, then exclude
    no_aspect_ids = sorted(df[df["aspect_category"] == ""]["review_id"].unique())
    df = df[df["aspect_category"] != ""].copy()

    # 3) malformed / column-shifted rows: aspect_category not one of the 7
    bad = ~df["aspect_category"].isin(VALID_CATS)
    n_bad = int(bad.sum())
    bad_examples = df.loc[bad, ["review_id", "aspect_category"]].head(10).values.tolist()
    df = df[~bad].copy()

    # 4) normalize + validate sentiment
    df["sentiment"] = df["sentiment"].str.lower()
    n_bad_sent = int((~df["sentiment"].isin(VALID_SENT)).sum())

    # 5) drop exact-duplicate quads (dedup on the quad key, ignore review text/notes)
    key = ["review_id", "aspect_term", "aspect_category", "opinion_term", "sentiment"]
    before = len(df)
    df = df.drop_duplicates(subset=key, keep="first").reset_index(drop=True)
    n_dupe = before - len(df)

    df = df[COLS]
    df.to_csv(OUT, index=False, encoding="utf-8")

    # ---- report ----
    print(f"[clean] raw rows read:            {n_raw}")
    print(f"[clean] empty/trailing dropped:   {n_empty}")
    print(f"[clean] 'no aspects' reviews:     {len(no_aspect_ids)} {no_aspect_ids}")
    print(f"[clean] malformed rows dropped:   {n_bad}")
    if bad_examples:
        print(f"        e.g. {bad_examples}")
    print(f"[clean] exact-duplicate dropped:  {n_dupe}")
    print(f"[clean] invalid sentiment left:   {n_bad_sent}")
    print(f"\n[clean] wrote {OUT}")
    print(f"[clean] final: {len(df)} quads over {df['review_id'].nunique()} reviews")
    print("[clean] category distribution:")
    print(df["aspect_category"].value_counts().to_string())
    print("[clean] sentiment:", df["sentiment"].value_counts().to_dict())
    print(f"[clean] IMPLICIT aspect_term: {(df['aspect_term'].str.upper()=='IMPLICIT').sum()}"
          f" | opinion_term: {(df['opinion_term'].str.upper()=='IMPLICIT').sum()}")


if __name__ == "__main__":
    main()
