"""
Stage 2: Quad-ABSA extraction on ALL reviews with Mistral Nemo (free/local).

Reads data/output/absa_input.csv, extracts quadruples
(aspect_category, aspect_term, opinion_term, sentiment) per review using the
taxonomy in config.ASPECT_CATEGORIES, and writes Mistral_predicted_quadruples.csv.

Per-review checkpointed and resumable — an interrupted run continues where it
left off. On CPU this is a long run (~24-28 h for 1000 reviews).

    python stages/stage2_llm_extract.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OUTPUT, INTERIM  # noqa: E402
from stages.compare_absa_models import (  # noqa: E402
    build_system, build_extractor, parse_quads, MODELS,
)

MODEL_KEY = "mistral_nemo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(OUTPUT / "absa_input.csv"),
                    help="input CSV with review_id, facility_type_en, english_translated")
    ap.add_argument("--out", default=str(OUTPUT / "Mistral_predicted_quadruples.csv"))
    ap.add_argument("--ckpt", default=str(INTERIM / "Mistral_predicted_quadruples_partial.csv"))
    args = ap.parse_args()
    SRC, OUT, CKPT = Path(args.src), Path(args.out), Path(args.ckpt)

    df = pd.read_csv(SRC)
    system = build_system()
    spec = MODELS[MODEL_KEY]

    rows = []
    done_ids = set()
    if CKPT.exists():
        prev = pd.read_csv(CKPT)
        rows = prev.to_dict("records")
        done_ids = set(prev["review_id"].astype(str))
        print(f"[stage2] resume: {len(done_ids)} reviews already done")

    todo = df[~df["review_id"].astype(str).isin(done_ids)]
    print(f"[stage2] {len(todo)}/{len(df)} reviews to extract with {spec['name']}")
    extract = build_extractor(spec, system)

    n_err = 0
    for k, (_, r) in enumerate(todo.iterrows(), 1):
        rid = str(r["review_id"])
        t0 = time.time()
        try:
            quads, ok = parse_quads(extract(str(r["english_translated"])))
        except Exception as e:  # never lose the whole run to one bad review
            print(f"[stage2] ERROR on {rid}: {type(e).__name__}: {e}")
            quads, ok = [], False
        if not ok:
            n_err += 1
        base = {"review_id": rid, "facility_type_en": r.get("facility_type_en", ""),
                "english_translated": r["english_translated"]}
        if not quads:
            rows.append({**base, "aspect_category": "", "aspect_term": "",
                         "opinion_term": "", "sentiment": "", "parse_ok": ok})
        for q in quads:
            rows.append({**base, **q, "parse_ok": ok})
        pd.DataFrame(rows).to_csv(CKPT, index=False, encoding="utf-8")  # checkpoint
        print(f"[stage2] {k}/{len(todo)} {rid}: {len(quads)} quads ({time.time() - t0:.0f}s)",
              flush=True)

    final = pd.DataFrame(rows)
    order = ["review_id", "facility_type_en", "aspect_category", "aspect_term",
             "opinion_term", "sentiment", "parse_ok", "english_translated"]
    final = final[[c for c in order if c in final.columns]]
    final.to_csv(OUT, index=False, encoding="utf-8")

    quads = final[final["aspect_category"] != ""]
    print(f"\n[stage2] wrote {OUT}")
    print(f"[stage2] rows: {len(final)} | total quads: {len(quads)} | "
          f"reviews with >=1 quad: {quads['review_id'].nunique()} | parse errors: {n_err}")


if __name__ == "__main__":
    main()
