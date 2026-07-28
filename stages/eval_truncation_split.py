"""
Robustness check: do the model metrics / ranking survive dropping the 12 truncated
evaluation reviews (en/ru length ratio < 0.65)? Reports SAFE-F1 & exact-F1 per model
on the full 50, the clean 38, and the truncated 12, plus a review-level cluster
bootstrap of the ranking on the clean 38.

    python stages/eval_truncation_split.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "stages"))
import safe_eval as se  # noqa: E402
from bootstrap_ci import per_review_counts, f1_from_counts, load_human  # noqa: E402

MODELS = ("mistral", "chatgpt", "claude")
DISP = {"mistral": "Mistral-Nemo", "chatgpt": "GPT-5.5", "claude": "Claude"}
B = 5000
TRUNC = {"r003388", "r019372", "r019437", "r020810", "r024441", "r024809",
         "r035116", "r043742", "r049654", "r063687", "r065156", "r067341"}


def metrics(counts_by_model, ids, keep: set):
    """SAFE-F1 / exact-F1 per model over reviews in `keep` (subset of ids)."""
    mask = np.array([i for i, r in enumerate(ids) if r in keep])
    out = {}
    for m in MODELS:
        s = counts_by_model[m][mask].sum(0)
        out[m] = (f1_from_counts(s[0], s[1], s[2]), f1_from_counts(s[3], s[4], s[5]))
    return out


def main():
    ids, human = load_human(ROOT / "data/output/human_annotation_template_50.xlsx")
    idset = set(ids)
    from sentence_transformers import SentenceTransformer
    print(f"[trunc] loading {se.MODEL_NAME} ...")
    se._model = SentenceTransformer(se.MODEL_NAME)

    counts = {m: per_review_counts(human, se.load_quads(se.FILES[m], idset), ids) for m in MODELS}
    clean = idset - TRUNC
    trunc = idset & TRUNC
    print(f"[trunc] full={len(idset)}  clean={len(clean)}  truncated={len(trunc)}")

    full_m = metrics(counts, ids, idset)
    clean_m = metrics(counts, ids, clean)
    trunc_m = metrics(counts, ids, trunc)

    print("\n=== SAFE-F1 (exact-F1) by split ===")
    print(f"{'model':14s} {'full 50':>16s} {'clean 38':>16s} {'truncated 12':>16s}")
    for m in MODELS:
        f = full_m[m]; c = clean_m[m]; t = trunc_m[m]
        print(f"{DISP[m]:14s} {f'{f[0]:.1f} ({f[1]:.1f})':>16s} "
              f"{f'{c[0]:.1f} ({c[1]:.1f})':>16s} {f'{t[0]:.1f} ({t[1]:.1f})':>16s}")

    def order(md):
        return " > ".join(DISP[m] for m in sorted(MODELS, key=lambda x: -md[x][0]))
    print(f"\nSAFE-F1 ranking  full : {order(full_m)}")
    print(f"SAFE-F1 ranking  clean: {order(clean_m)}")

    # ---- bootstrap the ranking on the clean 38 ----
    clean_idx = np.array([i for i, r in enumerate(ids) if r in clean])
    nC = len(clean_idx)
    rng = np.random.default_rng(42)
    boot = {m: np.empty(B) for m in MODELS}
    for b in range(B):
        pick = clean_idx[rng.integers(0, nC, size=nC)]
        for m in MODELS:
            s = counts[m][pick].sum(0)
            boot[m][b] = f1_from_counts(s[0], s[1], s[2])
    print(f"\n=== Bootstrap on CLEAN 38 (B={B}) ===")
    for m in MODELS:
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        print(f"  {DISP[m]:14s} SAFE-F1 {clean_m[m][0]:5.1f}  95% CI [{lo:5.1f}, {hi:5.1f}]")
    dmg = boot["mistral"] - np.maximum(boot["chatgpt"], boot["claude"])
    print(f"  P(Mistral is best on clean 38) = {(np.argmax(np.vstack([boot[m] for m in MODELS]),0)==0).mean():.3f}")
    print(f"  Mistral - best-other: median {np.median(dmg):+.1f}  95% CI "
          f"[{np.percentile(dmg,2.5):+.1f}, {np.percentile(dmg,97.5):+.1f}]")
    dcg = boot["claude"] - boot["chatgpt"]
    print(f"  P(Claude > GPT-5.5 on clean 38) = {(dcg>0).mean():.3f}  "
          f"(Δ median {np.median(dcg):+.1f})")

    rows = []
    for m in MODELS:
        rows.append({"model": DISP[m],
                     "SAFE_full": round(full_m[m][0], 1), "exact_full": round(full_m[m][1], 1),
                     "SAFE_clean38": round(clean_m[m][0], 1), "exact_clean38": round(clean_m[m][1], 1),
                     "SAFE_trunc12": round(trunc_m[m][0], 1), "exact_trunc12": round(trunc_m[m][1], 1)})
    pd.DataFrame(rows).to_csv(ROOT / "reports/truncation_split.csv", index=False, encoding="utf-8")
    print(f"\n[trunc] wrote {ROOT/'reports/truncation_split.csv'}")


if __name__ == "__main__":
    main()
