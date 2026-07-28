"""
Cluster bootstrap over the 50 eval reviews for SAFE-F1 / exact-F1.

SAFE (and exact) F1 is a micro-F1: TP/FP/FN are summed across reviews, and the
greedy 1-to-1 matching is done WITHIN each review independently. So each review's
(tp, fp, fn) is fixed regardless of which other reviews are in the set -> we can
compute per-review counts once, then bootstrap by resampling reviews (with
replacement) and re-summing. This gives exact percentile CIs cheaply and also a
paired "probability of superiority" for the model ordering.

    python stages/bootstrap_ci.py            # B=5000, seed 42
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "stages"))
import safe_eval as se  # noqa: E402

TEMPLATE = ROOT / "data/output/human_annotation_template_50.xlsx"
MODELS = ("mistral", "chatgpt", "claude")
DISP = {"mistral": "Mistral-Nemo", "chatgpt": "GPT-5.5", "claude": "Claude"}
B = 5000
TAU = 0.70


def load_human(src: Path):
    d = pd.read_excel(src, "annotation", dtype=str).fillna("")
    d = d[d["aspect_category"].astype(str).str.strip() != ""]
    ids = sorted(d["review_id"].astype(str).unique())
    tup = {r: [] for r in ids}
    for _, row in d.iterrows():
        r = str(row["review_id"])
        tup[r].append(tuple(str(row[c]).strip().lower() for c in se.QUAD))
    return ids, tup


def per_review_counts(ref: dict, pred: dict, ids: list[str]):
    """Return arrays of per-review (safe_tp,safe_fp,safe_fn, ex_tp,ex_fp,ex_fn)."""
    rows = []
    for rid in ids:
        A, P = ref.get(rid, []), pred.get(rid, [])
        # SAFE
        used = set(); stp = sfp = 0
        for p in P:
            best, best_i = 0.0, -1
            for i, a in enumerate(A):
                if i in used:
                    continue
                s = se.quad_sim(a, p) / 100.0
                if s > best:
                    best, best_i = s, i
            if best >= TAU:
                stp += 1; used.add(best_i)
            else:
                sfp += 1
        sfn = len(A) - len(used)
        # EXACT
        usedx = set(); etp = efp = 0
        for p in P:
            m = next((i for i, a in enumerate(A) if i not in usedx and a == p), None)
            if m is None:
                efp += 1
            else:
                etp += 1; usedx.add(m)
        efn = len(A) - len(usedx)
        rows.append((stp, sfp, sfn, etp, efp, efn))
    return np.array(rows, dtype=float)


def f1_from_counts(tp, fp, fn):
    pr = tp / (tp + fp) if (tp + fp) else 0.0
    rc = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * pr * rc / (pr + rc) * 100 if (pr + rc) else 0.0


def main():
    ids, human = load_human(TEMPLATE)
    idset = set(ids)
    n = len(ids)
    from sentence_transformers import SentenceTransformer
    print(f"[boot] loading {se.MODEL_NAME} ...")
    se._model = SentenceTransformer(se.MODEL_NAME)

    counts = {}
    for m in MODELS:
        pred = se.load_quads(se.FILES[m], idset)
        counts[m] = per_review_counts(human, pred, ids)

    # point estimates (full set)
    point = {}
    for m in MODELS:
        c = counts[m].sum(0)
        point[m] = {"SAFE_F1": f1_from_counts(c[0], c[1], c[2]),
                    "exact_F1": f1_from_counts(c[3], c[4], c[5])}

    rng = np.random.default_rng(42)
    idx = rng.integers(0, n, size=(B, n))            # B bootstrap resamples of review indices

    boot = {m: {"SAFE_F1": np.empty(B), "exact_F1": np.empty(B)} for m in MODELS}
    for m in MODELS:
        C = counts[m]
        for b in range(B):
            s = C[idx[b]].sum(0)
            boot[m]["SAFE_F1"][b] = f1_from_counts(s[0], s[1], s[2])
            boot[m]["exact_F1"][b] = f1_from_counts(s[3], s[4], s[5])

    def ci(a):
        return np.percentile(a, 2.5), np.percentile(a, 97.5)

    print(f"\n=== Bootstrap 95% CIs (B={B}, cluster=review, n={n}) ===")
    print(f"{'model':14s} {'SAFE-F1':>8s}  {'95% CI':>16s}   {'exact-F1':>8s}  {'95% CI':>16s}")
    for m in MODELS:
        s_lo, s_hi = ci(boot[m]["SAFE_F1"]); e_lo, e_hi = ci(boot[m]["exact_F1"])
        print(f"{DISP[m]:14s} {point[m]['SAFE_F1']:8.1f}  [{s_lo:5.1f}, {s_hi:5.1f}]   "
              f"{point[m]['exact_F1']:8.1f}  [{e_lo:5.1f}, {e_hi:5.1f}]")

    # paired superiority
    print(f"\n=== Paired bootstrap: P(A > B) on SAFE-F1 (paired over the same resamples) ===")
    pairs = [("mistral", "chatgpt"), ("mistral", "claude"), ("claude", "chatgpt")]
    for a, b in pairs:
        da = boot[a]["SAFE_F1"] - boot[b]["SAFE_F1"]
        p_ab = (da > 0).mean()
        d_lo, d_hi = ci(da)
        print(f"  P({DISP[a]} > {DISP[b]}) = {p_ab:5.3f}   "
              f"Δ median {np.median(da):+5.1f}  95% CI [{d_lo:+5.1f}, {d_hi:+5.1f}]")

    # P(Mistral is the single best on SAFE-F1)
    stack = np.vstack([boot[m]["SAFE_F1"] for m in MODELS])   # 3 x B
    best = np.argmax(stack, 0)
    print(f"\n=== P(model is the single best on SAFE-F1) ===")
    for i, m in enumerate(MODELS):
        print(f"  {DISP[m]:14s} {(best == i).mean():5.3f}")

    # save
    out = ROOT / "reports/bootstrap_ci.csv"
    recs = []
    for m in MODELS:
        s_lo, s_hi = ci(boot[m]["SAFE_F1"]); e_lo, e_hi = ci(boot[m]["exact_F1"])
        recs.append({"model": DISP[m], "SAFE_F1": round(point[m]["SAFE_F1"], 1),
                     "SAFE_F1_lo": round(s_lo, 1), "SAFE_F1_hi": round(s_hi, 1),
                     "exact_F1": round(point[m]["exact_F1"], 1),
                     "exact_F1_lo": round(e_lo, 1), "exact_F1_hi": round(e_hi, 1)})
    pd.DataFrame(recs).to_csv(out, index=False, encoding="utf-8")
    print(f"\n[boot] wrote {out}")


if __name__ == "__main__":
    main()
