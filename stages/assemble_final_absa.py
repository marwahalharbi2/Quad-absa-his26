"""
Assemble the final clean 1000-review Quad-ABSA dataset.

Combine successfully-extracted reviews from the original 1000 with successfully-
extracted reviews from the 200-review top-up, dropping the 140 that parse-failed,
to reach exactly TARGET reviews that all have >= 1 quadruple. Also remap any
off-taxonomy aspect categories the model invented to "Other".

    python stages/assemble_final_absa.py

Output: data/output/predicted_quadruples_final.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ASPECT_CATEGORIES, OUTPUT  # noqa: E402

ORIG = OUTPUT / "Mistral_predicted_quadruples.csv"
NEW = OUTPUT / "Mistral_predicted_quadruples_new200.csv"
OUT = OUTPUT / "Mistral_predicted_quadruples_final.csv"
TARGET = 1000
VALID = set(ASPECT_CATEGORIES.keys())


def good_reviews(df: pd.DataFrame) -> list[str]:
    """review_ids that have >= 1 real quad (non-empty aspect_category)."""
    has = df[df["aspect_category"].notna() &
             (df["aspect_category"].astype(str).str.strip() != "")]
    # preserve first-seen order
    seen, out = set(), []
    for rid in has["review_id"].astype(str):
        if rid not in seen:
            seen.add(rid)
            out.append(rid)
    return out


def main():
    orig = pd.read_csv(ORIG)
    new = pd.read_csv(NEW)

    good_orig = good_reviews(orig)
    good_new = good_reviews(new)
    print(f"[assemble] good original reviews: {len(good_orig)}")
    print(f"[assemble] good new reviews:      {len(good_new)}")

    need = TARGET - len(good_orig)
    if need < 0:
        keep_orig, keep_new = good_orig[:TARGET], []
    else:
        keep_orig, keep_new = good_orig, good_new[:need]
    if len(keep_orig) + len(keep_new) < TARGET:
        print(f"[assemble] WARNING: only {len(keep_orig) + len(keep_new)} good "
              f"reviews available (< {TARGET}); adding all new good ones")
        keep_new = good_new

    keep_ids = set(keep_orig) | set(keep_new)
    parts = [
        orig[orig["review_id"].astype(str).isin(set(keep_orig))],
        new[new["review_id"].astype(str).isin(set(keep_new))],
    ]
    final = pd.concat(parts, ignore_index=True)
    # keep only real quad rows (drop empty placeholder rows)
    final = final[final["aspect_category"].notna() &
                  (final["aspect_category"].astype(str).str.strip() != "")].copy()

    # remap off-taxonomy categories -> Other
    off = ~final["aspect_category"].isin(VALID)
    n_off = int(off.sum())
    if n_off:
        print(f"[assemble] remapping {n_off} off-taxonomy category rows -> Other: "
              f"{sorted(final.loc[off, 'aspect_category'].unique())}")
        final.loc[off, "aspect_category"] = "Other"

    cols = ["review_id", "facility_type_en", "aspect_category", "aspect_term",
            "opinion_term", "sentiment", "english_translated"]
    final = final[[c for c in cols if c in final.columns]]
    final.to_csv(OUT, index=False, encoding="utf-8")

    n_reviews = final["review_id"].nunique()
    print(f"\n[assemble] wrote {OUT}")
    print(f"[assemble] reviews: {n_reviews} | quads: {len(final)} "
          f"(from {len(keep_orig)} original + {len(keep_new)} new)")
    print("[assemble] category distribution:")
    print(final["aspect_category"].value_counts().to_string())
    print("[assemble] sentiment:", final["sentiment"].value_counts().to_dict())


if __name__ == "__main__":
    main()
