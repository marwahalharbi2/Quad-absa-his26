"""
Select the fixed evaluation set of 50 reviews for Stage 2 (model comparison).

Complete-cases design: the 50 must be **fully annotated by every LLM** present
(Mistral / ChatGPT / Claude). Selection is RANDOM but covers all facility types
present in that intersection (>=1 per type, then random-fill to 50, seed=42).

The human then annotates exactly these 50 to form the gold standard.

    python stages/select_eval_50.py            # uses whatever *_100_annotation.csv exist
    python stages/select_eval_50.py --n 50

Output: data/output/eval_50_ids.csv  (review_id, facility_type_en)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OUTPUT  # noqa: E402

SAMPLE = OUTPUT / "sample_100.csv"
OUT = OUTPUT / "eval_50_ids.csv"
CANDIDATES = {  # model -> annotation file
    "mistral": OUTPUT / "mistral_100_annotation.csv",
    "chatgpt": OUTPUT / "chatgpt_100_annotation.csv",
    "claude":  OUTPUT / "claude_100_annotation.csv",
}
SEED = 42


def covered(path: Path) -> set[str]:
    """review_ids with >=1 real quad in an annotation file."""
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    q = d[d["aspect_category"].astype(str).str.strip() != ""]
    return set(q["review_id"].astype(str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    present = {m: p for m, p in CANDIDATES.items() if p.exists()}
    print(f"[select] models present: {list(present)}")
    missing = [m for m in CANDIDATES if m not in present]
    if missing:
        print(f"[select] NOT YET available (dry-run intersection only): {missing}")

    inter = None
    for m, p in present.items():
        c = covered(p)
        inter = c if inter is None else (inter & c)
        print(f"[select]   {m}: {len(c)} fully-annotated reviews")
    print(f"[select] intersection (all present models complete): {len(inter)}")

    # attach facility type
    sample = pd.read_csv(SAMPLE, dtype=str)[["review_id", "facility_type_en"]]
    pool = sample[sample["review_id"].astype(str).isin(inter)].reset_index(drop=True)
    types = sorted(pool["facility_type_en"].dropna().unique())
    print(f"[select] facility types in the pool: {len(types)}")

    n = args.n
    if len(pool) <= n:
        chosen = pool.copy()
        print(f"[select] pool ({len(pool)}) <= n ({n}); taking ALL of it")
    else:
        # 1 random per facility type (coverage), then random-fill to n
        first = pool.groupby("facility_type_en", group_keys=False).apply(
            lambda g: g.sample(1, random_state=SEED))
        remaining = pool[~pool["review_id"].isin(first["review_id"])]
        need = n - len(first)
        fill = remaining.sample(min(need, len(remaining)), random_state=SEED) if need > 0 else remaining.iloc[0:0]
        chosen = pd.concat([first, fill]).sort_values("facility_type_en").reset_index(drop=True)

    chosen = chosen[["review_id", "facility_type_en"]]
    chosen.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\n[select] wrote {OUT}: {len(chosen)} reviews")
    print(f"[select] facility types covered: {chosen['facility_type_en'].nunique()} / {len(types)}")
    print(f"[select] all pool types present in selection? "
          f"{set(chosen['facility_type_en']) == set(types)}")
    if missing:
        print(f"[select] NOTE: dry-run set (missing {missing}); re-run once Claude is in "
              f"to lock the true 3-way 50.")


if __name__ == "__main__":
    main()
