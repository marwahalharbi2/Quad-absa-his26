"""
Partial / interim Stage-2 test — score the three LLMs against however many of the
50 eval reviews the human has annotated SO FAR.

Reads the human annotations straight from the fill-in template
(`data/output/human_annotation_template_50.xlsx`, sheet "annotation") — so you can
re-run it any time you add more rows — or from a CSV via --src. Evaluation is scoped
to ONLY the reviews the human has actually annotated, so the unannotated eval reviews
don't get counted as false positives.

Reuses stages/safe_eval.py (SAFE, MiniLM, tau=0.70) — concept unchanged.

    python stages/eval_human_partial.py                 # read the xlsx template
    python stages/eval_human_partial.py --src data/output/human_100_annotation.csv
    python stages/eval_human_partial.py --tau 0.75

Outputs (reports/):
    partial_human_summary.csv     one row per model (exact-F1, SAFE-F1, P/R, ...)
    partial_human_by_category.csv aspect_category x model recall (which categories get missed)
    partial_human_by_review.csv   review x model matched/missed + the missed human spans
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stages"))
import safe_eval as se  # noqa: E402
from config import OUTPUT, REPORTS  # noqa: E402

TEMPLATE = OUTPUT / "human_annotation_template_50.xlsx"
MODELS = ("mistral", "chatgpt", "claude")


def load_human(src: Path):
    """Return (ids, human_tuples{rid:[quad tuples]}, human_disp{rid:[(term,cat,op,sent)]})."""
    if src.suffix.lower() in (".xlsx", ".xlsm"):
        d = pd.read_excel(src, "annotation", dtype=str).fillna("")
    else:
        d = pd.read_csv(src, dtype=str, keep_default_na=False)
    d = d[d["aspect_category"].astype(str).str.strip() != ""]
    ids = sorted(d["review_id"].astype(str).unique())
    tup, disp = {r: [] for r in ids}, {r: [] for r in ids}
    for _, row in d.iterrows():
        r = str(row["review_id"])
        tup[r].append(tuple(str(row[c]).strip().lower() for c in se.QUAD))
        disp[r].append(tuple(str(row[c]).strip() for c in se.QUAD))
    return ids, tup, disp


def matched_human_idx(A: list, P: list, tau: float) -> set:
    """Greedy 1-to-1 (same as safe_eval); return indices of human quads that got matched."""
    used = set()
    for p in P:
        best, best_i = 0.0, -1
        for i, a in enumerate(A):
            if i in used:
                continue
            s = se.quad_sim(a, p) / 100.0
            if s > best:
                best, best_i = s, i
        if best >= tau and best_i >= 0:
            used.add(best_i)
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(TEMPLATE), help="human annotations (xlsx template or csv)")
    ap.add_argument("--tau", type=float, default=se.TAU)
    args = ap.parse_args()
    se.TAU = args.tau

    src = Path(args.src)
    ids, human, disp = load_human(src)
    idset = set(ids)
    n_quads = sum(len(v) for v in human.values())
    print(f"[partial] source: {src.name}")
    print(f"[partial] human-annotated reviews: {len(ids)} | human quads: {n_quads} | tau={args.tau}")
    if not ids:
        sys.exit("[partial] no human quads found — fill some rows first.")

    from sentence_transformers import SentenceTransformer
    print(f"[partial] loading {se.MODEL_NAME} ...")
    se._model = SentenceTransformer(se.MODEL_NAME)

    preds = {m: se.load_quads(se.FILES[m], idset) for m in MODELS if se.FILES[m].exists()}

    # ---- 1) summary table ----
    summ = []
    for m, pred in preds.items():
        met = se.score(human, pred, ids)
        covered = sum(1 for r in ids if pred.get(r))
        summ.append({"model": m, "human_reviews": len(ids), "llm_covered": covered, **met})
    summ_df = pd.DataFrame(summ)
    print("\n=== SUMMARY (per model, vs human, scoped to annotated reviews) ===")
    for r in summ:
        print(f"  {r['model']:8s} | covered {r['llm_covered']}/{len(ids)} | "
              f"exact-F1 {r['exact_F1']:5.1f} | SAFE-F1 {r['SAFE_F1']:5.1f} | "
              f"mean-SAFE {r['mean_SAFE']:5.1f} | SAFE P/R {r['SAFE_P']:.0f}/{r['SAFE_R']:.0f}")

    # ---- per-review matched sets (reused by 2 and 3) ----
    match = {m: {r: matched_human_idx(human[r], preds[m].get(r, []), args.tau) for r in ids}
             for m in preds}

    # ---- 2) per-category recall ----
    cat_rows = []
    cats = sorted({disp[r][i][1] for r in ids for i in range(len(disp[r]))})
    for cat in cats:
        row = {"aspect_category": cat}
        for m in preds:
            tot = matched = 0
            for r in ids:
                for i, q in enumerate(disp[r]):
                    if q[1] == cat:
                        tot += 1
                        matched += (i in match[m][r])
            row[f"{m}_matched"] = matched
            row[f"{m}_total"] = tot
            row[f"{m}_recall%"] = round(100 * matched / tot, 0) if tot else 0
        cat_rows.append(row)
    cat_df = pd.DataFrame(cat_rows)
    print("\n=== PER-CATEGORY recall (human quads found, by model) ===")
    hdr = "  {:<28s}".format("aspect_category") + "".join(f"{m[:7]:>12s}" for m in preds)
    print(hdr)
    for row in cat_rows:
        line = "  {:<28s}".format(row["aspect_category"])
        for m in preds:
            line += f"{row[f'{m}_matched']}/{row[f'{m}_total']} ({row[f'{m}_recall%']:.0f}%)".rjust(12)
        print(line)

    # ---- 3) per-review misses ----
    rev_rows = []
    print("\n=== PER-REVIEW: what each model MISSED (human quads not matched) ===")
    for r in ids:
        print(f"  {r}  ({len(human[r])} human quads)")
        for m in preds:
            missed = [i for i in range(len(human[r])) if i not in match[m][r]]
            miss_terms = "; ".join(f"{disp[r][i][0]} [{disp[r][i][1]}/{disp[r][i][3]}]" for i in missed)
            rev_rows.append({"review_id": r, "model": m, "human_quads": len(human[r]),
                             "matched": len(human[r]) - len(missed), "missed": len(missed),
                             "missed_spans": miss_terms})
            tag = "OK" if not missed else f"missed {len(missed)}"
            print(f"      {m:8s} {len(human[r])-len(missed)}/{len(human[r])} matched"
                  + (f"  ->  {miss_terms}" if missed else "  (all found)"))

    REPORTS.mkdir(exist_ok=True)
    summ_df.to_csv(REPORTS / "partial_human_summary.csv", index=False, encoding="utf-8")
    cat_df.to_csv(REPORTS / "partial_human_by_category.csv", index=False, encoding="utf-8")
    pd.DataFrame(rev_rows).to_csv(REPORTS / "partial_human_by_review.csv", index=False, encoding="utf-8")
    print(f"\n[partial] wrote 3 reports to {REPORTS}/partial_human_*.csv")


if __name__ == "__main__":
    main()
