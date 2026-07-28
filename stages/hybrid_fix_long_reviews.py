"""
Hybrid fix for Stage -1: opus_tc_big produces repetition-degenerate output on
long reviews (flagged via `repetition_flag`). Re-translate ONLY those reviews
with Mistral Nemo (best quality on the benchmark, handles long reviews cleanly),
then merge into a clean 1000-review dataset.

Per-review checkpointed (~6-8 hrs on CPU), so an interruption resumes.

    python stages/hybrid_fix_long_reviews.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OUTPUT, INTERIM  # noqa: E402
from stages.stage_m1_translate import resolve_device, is_degenerate  # noqa: E402
from stages.compare_translation_models import MistralNemoModel  # noqa: E402

SRC_CSV = OUTPUT / "translations_opus1000.csv"
OUT_CSV = OUTPUT / "translations_opus1000_hybrid.csv"
CKPT = INTERIM / "hybrid_fix_partial.csv"
MAX_TOKENS = 1536          # long reviews need room for a full translation
N_CTX = 8192               # fit long Russian input + output


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(("true", "1"))


def main():
    df = pd.read_csv(SRC_CSV)
    flagged = df[as_bool(df["repetition_flag"])]
    print(f"[hybrid] {len(flagged)} flagged long reviews to re-translate with Mistral Nemo")
    print(f"[hybrid] their ru length: min={int(flagged['original_russian'].str.len().min())} "
          f"mean={int(flagged['original_russian'].str.len().mean())} "
          f"max={int(flagged['original_russian'].str.len().max())}")

    done: dict[str, str] = {}
    if CKPT.exists():
        prev = pd.read_csv(CKPT)
        done = dict(zip(prev["review_id"].astype(str), prev["english_translated"]))
        print(f"[hybrid] resume: {len(done)} already re-translated")

    device = resolve_device("auto")
    MistralNemoModel.n_ctx = N_CTX
    print("[hybrid] loading Mistral Nemo ...")
    model = MistralNemoModel(device, MAX_TOKENS)

    todo = flagged[~flagged["review_id"].astype(str).isin(done)]
    print(f"[hybrid] {len(todo)} still to do")
    for k, (_, row) in enumerate(todo.iterrows(), 1):
        rid = str(row["review_id"])
        t0 = time.time()
        prompt = model.build_prompt(row["original_russian"], "ru", "en")
        r = model.llm(prompt, max_tokens=MAX_TOKENS, temperature=0.1, stop=model.stop)
        en = model.clean(r["choices"][0]["text"])
        done[rid] = en
        # checkpoint after EVERY review (each takes minutes)
        pd.DataFrame({"review_id": list(done), "english_translated": list(done.values())}
                     ).to_csv(CKPT, index=False, encoding="utf-8")
        print(f"[hybrid] {k}/{len(todo)} {rid} ({time.time() - t0:.0f}s) "
              f"still_degenerate={is_degenerate(en)}", flush=True)

    # ---- merge fixed translations back into the full dataset ----
    fixed = set(done)
    df["english_translated"] = df.apply(
        lambda r: done.get(str(r["review_id"]), r["english_translated"]), axis=1)
    df["translation_tool"] = df.apply(
        lambda r: "Mistral-Nemo-Instruct-2407 (hybrid fix)"
        if str(r["review_id"]) in fixed else r["translation_tool"], axis=1)
    df["repetition_flag"] = df["english_translated"].map(is_degenerate)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"\n[hybrid] wrote {OUT_CSV}")
    print(f"[hybrid] re-translated {len(fixed)} reviews with Mistral Nemo")
    print(f"[hybrid] remaining flagged after fix: {int(df['repetition_flag'].sum())}")


if __name__ == "__main__":
    main()
