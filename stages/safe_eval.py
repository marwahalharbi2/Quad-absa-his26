"""
Stage 2 - SAFE + exact-match F1 evaluation (adapted from safe_evaluation_merged.py).

Concept unchanged: SAFE = Semantic-Aware Flexible Evaluation. Each predicted quad
is matched to the best reference quad by semantic similarity (MiniLM embeddings);
best >= tau -> TP, unmatched prediction -> FP, unmatched reference -> FN.
Alongside SAFE we also report the strict EXACT-MATCH F1 baseline.

- Embedder : all-MiniLM-L6-v2   (single model)
- Threshold: tau = 70  (i.e. 0.70 on the 0-1 scale; SAFE score is 0-100)
- Weights  : equal 0.25 per element (Soft SAFE)
- Quad order: aspect -> category -> opinion -> sentiment
- Reference = human gold; prediction = each LLM (mistral / chatgpt / claude)
- Eval set  = the fixed ids in data/output/eval_50_ids.csv

    python stages/safe_eval.py                 # human gold vs each LLM (real run)
    python stages/safe_eval.py --ref chatgpt   # dry-run: use an LLM as the reference

Outputs: reports/safe_eval.csv (comparison table) + reports/safe_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OUTPUT, REPORTS  # noqa: E402

IDS = OUTPUT / "eval_50_ids.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TAU = 0.70                       # SAFE match threshold (0-1); score reported x100
QUAD = ["aspect_term", "aspect_category", "opinion_term", "sentiment"]  # aspect-first
FILES = {
    "human":   OUTPUT / "human_100_annotation.csv",
    "mistral": OUTPUT / "mistral_100_annotation.csv",
    "chatgpt": OUTPUT / "chatgpt_100_annotation.csv",
    "claude":  OUTPUT / "claude_100_annotation.csv",
}

_model = None
_cache: dict[str, object] = {}


def _emb(text: str):
    from sentence_transformers import util  # noqa: F401  (import kept local)
    if text not in _cache:
        _cache[text] = _model.encode(text, convert_to_tensor=True)
    return _cache[text]


def sim(x: str, y: str) -> float:
    """Semantic similarity, exact-match -> 1.0, else rescaled cosine in [0,1]."""
    from sentence_transformers import util
    x = str(x).strip().lower()
    y = str(y).strip().lower()
    if x == y:
        return 1.0
    if not x or not y:
        return 0.0
    return (util.cos_sim(_emb(x), _emb(y)).item() + 1) / 2.0


def quad_sim(a: tuple, p: tuple) -> float:
    """Soft SAFE score between two quads (mean of 4 element sims), 0-100."""
    return sum(sim(ai, pi) for ai, pi in zip(a, p)) / len(a) * 100.0


def load_quads(path: Path, ids: set[str]) -> dict[str, list[tuple]]:
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    d = d[d["review_id"].astype(str).isin(ids)]
    d = d[d["aspect_category"].astype(str).str.strip() != ""]
    out: dict[str, list[tuple]] = {r: [] for r in ids}
    for _, row in d.iterrows():
        out[str(row["review_id"])].append(tuple(str(row[c]).strip().lower() for c in QUAD))
    return out


def score(ref: dict, pred: dict, ids: list[str]) -> dict:
    """Compute SAFE (semantic, tau) and EXACT (strict) metrics over the id set."""
    safe_tp = safe_fp = safe_fn = 0
    ex_tp = ex_fp = ex_fn = 0
    review_safe = []

    for rid in ids:
        A, P = ref.get(rid, []), pred.get(rid, [])

        # ---- SAFE (semantic, greedy 1-to-1, threshold) ----
        used = set()
        best_per_pred = []
        for p in P:
            best, best_i = 0.0, -1
            for i, a in enumerate(A):
                if i in used:
                    continue
                s = quad_sim(a, p) / 100.0
                if s > best:
                    best, best_i = s, i
            best_per_pred.append(best * 100)
            if best >= TAU:
                safe_tp += 1
                used.add(best_i)
            else:
                safe_fp += 1
        safe_fn += len(A) - len(used)
        if P:
            review_safe.append(sum(best_per_pred) / len(best_per_pred))

        # ---- EXACT match (strict tuple equality, greedy 1-to-1) ----
        usedx = set()
        for p in P:
            m = next((i for i, a in enumerate(A) if i not in usedx and a == p), None)
            if m is None:
                ex_fp += 1
            else:
                ex_tp += 1
                usedx.add(m)
        ex_fn += len(A) - len(usedx)

    def prf(tp, fp, fn):
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        return round(pr * 100, 2), round(rc * 100, 2), round(f1 * 100, 2)

    sp, sr, sf1 = prf(safe_tp, safe_fp, safe_fn)
    xp, xr, xf1 = prf(ex_tp, ex_fp, ex_fn)
    return {
        "mean_SAFE": round(sum(review_safe) / len(review_safe), 2) if review_safe else 0.0,
        "SAFE_P": sp, "SAFE_R": sr, "SAFE_F1": sf1,
        "SAFE_TP": safe_tp, "SAFE_FP": safe_fp, "SAFE_FN": safe_fn,
        "exact_P": xp, "exact_R": xr, "exact_F1": xf1,
        "exact_TP": ex_tp, "exact_FP": ex_fp, "exact_FN": ex_fn,
    }


def main():
    global _model
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="human", choices=list(FILES),
                    help="reference/gold source (default: human)")
    args = ap.parse_args()

    if not IDS.exists():
        sys.exit("[safe] missing eval set: run stages/select_eval_50.py first")
    ids = [str(x) for x in pd.read_csv(IDS)["review_id"]]
    idset = set(ids)

    ref_name = args.ref
    if not FILES[ref_name].exists():
        sys.exit(f"[safe] reference '{ref_name}' not found: {FILES[ref_name]}")
    preds = [m for m in ("mistral", "chatgpt", "claude")
             if m != ref_name and FILES[m].exists()]
    print(f"[safe] reference = {ref_name} | predictions = {preds} | eval reviews = {len(ids)}")
    if ref_name != "human":
        print("[safe] DRY-RUN: using an LLM as the reference (human gold not in yet)")

    from sentence_transformers import SentenceTransformer
    print(f"[safe] loading {MODEL_NAME} ...")
    _model = SentenceTransformer(MODEL_NAME)

    ref = load_quads(FILES[ref_name], idset)
    rows = []
    for m in preds:
        met = score(ref, load_quads(FILES[m], idset), ids)
        met = {"model": m, "vs_reference": ref_name, **met}
        rows.append(met)
        print(f"[safe] {m:8s} | exact-F1 {met['exact_F1']:5.1f} | "
              f"SAFE-F1 {met['SAFE_F1']:5.1f} | mean-SAFE {met['mean_SAFE']:5.1f}")

    REPORTS.mkdir(exist_ok=True)
    out_csv = REPORTS / "safe_eval.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    (REPORTS / "safe_eval.json").write_text(
        json.dumps({"reference": ref_name, "tau": TAU, "n_reviews": len(ids),
                    "results": rows}, indent=2), encoding="utf-8")
    print(f"\n[safe] wrote {out_csv}")


if __name__ == "__main__":
    main()
