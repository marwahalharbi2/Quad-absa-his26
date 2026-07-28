"""
Stage -1: Load Russian healthcare-facility reviews and translate them to English.

Pipeline:
  1. Load `blinoff/healthcare_facilities_reviews` from Hugging Face.
  2. Auto-detect the review-text column, keep non-empty texts.
  3. Batch-translate ru -> en with M2M-100.
  4. Sample N reviews, back-translate en -> ru, score with multilingual BERTScore.
  5. Write translations CSV + a JSON quality report.

Run a small smoke test first:
    python stages/stage_m1_translate.py --limit 20 --sample 8

Then the full run:
    python stages/stage_m1_translate.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

# allow "python stages/stage_m1_translate.py" from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import STAGE_M1, RAW, INTERIM, OUTPUT, REPORTS  # noqa: E402


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def resolve_device(pref: str):
    import torch
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but not available; falling back to cpu")
        return "cpu"
    return pref


def detect_text_column(df: pd.DataFrame) -> str:
    """Pick the most likely free-text review column."""
    candidates = ["content", "text", "review", "review_text", "body", "comment"]
    for c in candidates:
        if c in df.columns:
            return c
    # fall back to the object column with the longest mean length
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if not obj_cols:
        raise ValueError(f"No text-like column found. Columns: {list(df.columns)}")
    lengths = {c: df[c].astype(str).str.len().mean() for c in obj_cols}
    return max(lengths, key=lengths.get)


def load_reviews(limit: int | None, stratify: bool = True, seed: int = 42) -> pd.DataFrame:
    # NOTE: this dataset ships a custom loader script, which `datasets` >=3.0
    # no longer supports. Pull the raw JSONL directly and read it with pandas.
    from huggingface_hub import hf_hub_download
    print(f"[stage-1] loading dataset {STAGE_M1['dataset_name']} ...")
    fp = hf_hub_download(
        STAGE_M1["dataset_name"],
        "healthcare_facilities_reviews.jsonl",
        repo_type="dataset",
    )
    df = pd.read_json(fp, lines=True)
    print(f"[stage-1] loaded: {len(df)} rows, columns={list(df.columns)}")

    text_col = detect_text_column(df)
    print(f"[stage-1] using text column: '{text_col}'")

    df = df.rename(columns={text_col: "original_russian"})
    df["original_russian"] = df["original_russian"].astype(str).str.strip()
    df = df[df["original_russian"].str.len() > 0].reset_index(drop=True)
    # the dataset's `category` is the FACILITY TYPE, not an ABSA aspect category
    if "category" in df.columns:
        df = df.rename(columns={"category": "facility_type"})
    # keep the dataset's own id as review_id_src, then add a stable padded id
    if "review_id" in df.columns:
        df = df.rename(columns={"review_id": "review_id_src"})
    df.insert(0, "review_id", [f"r{i:06d}" for i in range(len(df))])

    # carry ALL original metadata columns through the pipeline
    strat_col = "sentiment" if "sentiment" in df.columns else None

    if limit is not None and limit < len(df):
        if stratify and strat_col is not None:
            df = stratified_sample(df, limit, strat_col, seed)
            print(f"[stage-1] stratified sample of {len(df)} by '{strat_col}': "
                  f"{df[strat_col].value_counts().to_dict()}")
        else:
            df = df.head(limit).reset_index(drop=True)
    return df


def stratified_sample(df: pd.DataFrame, n: int, strat_col: str, seed: int) -> pd.DataFrame:
    """Proportional stratified sample of n rows by strat_col, sorted by review_id."""
    if n >= len(df):
        return df.sort_values("review_id").reset_index(drop=True)
    frac = n / len(df)
    out = (
        df.groupby(strat_col, group_keys=False)
        .apply(lambda g: g.sample(max(1, round(len(g) * frac)), random_state=seed))
    )
    out = out.sample(frac=1, random_state=seed).head(n)   # shuffle + trim to exactly n
    return out.sort_values("review_id").reset_index(drop=True)


class M2M100Translator:
    def __init__(self, model_name: str, device: str, max_input_tokens: int):
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        print(f"[stage-1] loading model {model_name} on {device} ...")
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.max_input_tokens = max_input_tokens

    def translate(self, texts: list[str], src: str, tgt: str, batch_size: int) -> list[str]:
        import torch
        self.tokenizer.src_lang = src
        forced_bos = self.tokenizer.get_lang_id(tgt)
        out: list[str] = []
        n = len(texts)
        for start in range(0, n, batch_size):
            batch = texts[start:start + batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_input_tokens,
            ).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc, forced_bos_token_id=forced_bos, max_length=self.max_input_tokens,
                )
            out.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
            done = min(start + batch_size, n)
            print(f"[stage-1]   translated {done}/{n}", end="\r", flush=True)
        print()
        return out


class _CompareModelAdapter:
    """Wrap a compare_translation_models adapter so it exposes the same
    translate(texts, src, tgt, batch_size) signature used by this pipeline,
    with the same in-place progress print."""

    def __init__(self, model, name):
        self.model = model
        self.name = name

    def translate(self, texts, src, tgt, batch_size):
        out = []
        n = len(texts)
        for start in range(0, n, batch_size):
            out.extend(self.model.translate(texts[start:start + batch_size],
                                            src, tgt, bs=batch_size))
            done = min(start + batch_size, n)
            print(f"[stage-1]   translated {done}/{n}", end="\r", flush=True)
        print()
        return out


def build_translator(model_key: str, device: str, max_len: int):
    """Return (translator, display_name). model_key='m2m100' uses the built-in
    M2M-100; any other key is resolved from compare_translation_models.REGISTRY."""
    if model_key == "m2m100":
        name = "facebook/m2m100_418M"
        return M2M100Translator(name, device, max_len), name
    from stages.compare_translation_models import REGISTRY
    if model_key not in REGISTRY:
        raise ValueError(f"unknown model '{model_key}'. options: "
                         f"m2m100,{','.join(REGISTRY)}")
    cls = REGISTRY[model_key]
    return _CompareModelAdapter(cls(device, max_len), cls.name), cls.name


def is_degenerate(text: str, min_tokens: int = 12, ratio: float = 0.45) -> bool:
    """Flag repetition-loop degeneration (e.g. opus-tc-big looping a phrase):
    long output with a low distinct-token ratio."""
    toks = str(text).split()
    if len(toks) < min_tokens:
        return False
    return len(set(toks)) / len(toks) < ratio


def translate_forward_checkpointed(translator, df: pd.DataFrame, src: str, tgt: str,
                                   batch_size: int, ckpt_path: Path,
                                   flush_every: int = 32) -> pd.Series:
    """Translate df['original_russian'] ru->en, saving progress to `ckpt_path`.

    Resumes automatically: any review_id already present in the checkpoint is
    skipped, so an interrupted run only loses the last unflushed chunk.
    """
    done: dict[str, str] = {}
    if ckpt_path.exists():
        prev = pd.read_csv(ckpt_path)
        done = dict(zip(prev["review_id"].astype(str), prev["english_translated"]))
        print(f"[stage-1] resume: {len(done)} translations found in checkpoint")

    todo = df[~df["review_id"].astype(str).isin(done)]
    remaining = list(zip(todo["review_id"].astype(str), todo["original_russian"]))
    print(f"[stage-1] {len(remaining)}/{len(df)} reviews still to translate")

    for start in range(0, len(remaining), flush_every):
        chunk = remaining[start:start + flush_every]
        ids = [c[0] for c in chunk]
        texts = [c[1] for c in chunk]
        translations = translator.translate(texts, src=src, tgt=tgt, batch_size=batch_size)
        done.update(zip(ids, translations))
        # persist the full accumulated checkpoint after every chunk
        pd.DataFrame({"review_id": list(done.keys()),
                      "english_translated": list(done.values())}).to_csv(
            ckpt_path, index=False, encoding="utf-8")
        print(f"[stage-1] checkpoint saved: {len(done)}/{len(df)} done")

    return df["review_id"].astype(str).map(done)


def _bertscore_f1(cands: list[str], refs: list[str], model: str) -> list[float]:
    from bert_score import score as bertscore
    P, R, F1 = bertscore(
        cands, refs, model_type=model, num_layers=9, verbose=False,
    )
    return [float(x) for x in F1]


def back_translation_quality(translator, df: pd.DataFrame, sample_size: int,
                             src: str, tgt: str, batch_size: int, bertscore_model: str,
                             seed: int, baseline_pairs: int) -> dict:
    """Back-translate en->ru on a sample and score vs the original Russian.

    `bert-score` ships no Russian rescale baseline, so we compute one
    empirically (Zhang et al. 2020): the mean BERTScore of random, unrelated
    Russian sentence pairs. Rescaled = (raw - baseline) / (1 - baseline), which
    maps random pairs -> ~0 and a perfect round-trip -> ~1, making the scores
    interpretable and a threshold defensible.
    """
    random.seed(seed)
    idx = list(df.index)
    sample_idx = random.sample(idx, min(sample_size, len(idx)))
    sample = df.loc[sample_idx]

    print(f"[stage-1] back-translating {len(sample)} samples for QA ...")
    back_ru = translator.translate(
        sample["english_translated"].tolist(), src=tgt, tgt=src, batch_size=batch_size,
    )
    raw = _bertscore_f1(back_ru, sample["original_russian"].tolist(), bertscore_model)

    # ---- empirical baseline: BERTScore of mismatched (unrelated) ru pairs ----
    originals = df["original_russian"].tolist()
    n_pairs = min(baseline_pairs, len(originals))
    pool = random.sample(originals, n_pairs) if n_pairs < len(originals) else list(originals)
    # pair each with the next (rotated) so no text is compared to itself
    shifted = pool[1:] + pool[:1]
    print(f"[stage-1] computing empirical ru baseline from {n_pairs} mismatched pairs ...")
    base_scores = _bertscore_f1(pool, shifted, bertscore_model)
    baseline = sum(base_scores) / len(base_scores) if base_scores else 0.0

    denom = (1.0 - baseline) or 1e-9
    rescaled = [(r - baseline) / denom for r in raw]
    return {
        "sample_idx": list(sample_idx),
        "raw_scores": raw,
        "rescaled_scores": rescaled,
        "baseline": baseline,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of reviews (smoke test)")
    ap.add_argument("--sample", type=int, default=None,
                    help="override validation sample size")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--tag", type=str, default="batch_1",
                    help="output filename suffix")
    ap.add_argument("--no-stratify", action="store_true",
                    help="take the first N reviews instead of a stratified sample")
    ap.add_argument("--model", type=str, default="m2m100",
                    help="translation model key: m2m100, opus, nllb, opus_tc_big, nllb_1.3B")
    ap.add_argument("--extend-from", type=str, default=None,
                    help="path to an existing translations CSV; keep its reviews as a "
                         "base and sample up to --limit total, only translating the new ones")
    args = ap.parse_args()

    p = dict(STAGE_M1)
    if args.sample is not None:
        p["sample_size_for_validation"] = args.sample
    if args.batch_size is not None:
        p["batch_size"] = args.batch_size

    device = resolve_device(p["device"])
    t0 = time.time()

    ckpt_path = INTERIM / f"translations_{args.tag}_partial.csv"

    if args.extend_from:
        # keep an existing set as base, add new stratified reviews up to --limit,
        # and pre-seed the checkpoint so only the new reviews get translated.
        base = pd.read_csv(args.extend_from)
        full = load_reviews(None, seed=p["seed"])
        base_ids = set(base["review_id"].astype(str))
        base_rows = full[full["review_id"].astype(str).isin(base_ids)]
        pool = full[~full["review_id"].astype(str).isin(base_ids)]
        n_new = max(0, args.limit - len(base_rows))
        new_rows = stratified_sample(pool, n_new, "sentiment", p["seed"] + 1)
        df = (pd.concat([base_rows, new_rows])
              .sort_values("review_id").reset_index(drop=True))
        print(f"[stage-1] extend: {len(base_rows)} base + {len(new_rows)} new "
              f"= {len(df)} total")
        # seed checkpoint with the base translations (skips re-translating them)
        base[["review_id", "english_translated"]].to_csv(ckpt_path, index=False,
                                                         encoding="utf-8")
    else:
        df = load_reviews(args.limit, stratify=not args.no_stratify, seed=p["seed"])
    print(f"[stage-1] translating {len(df)} reviews (batch_size={p['batch_size']}) ...")

    translator, model_name = build_translator(args.model, device, p["max_input_tokens"])
    print(f"[stage-1] translation model: {args.model} ({model_name})")
    df["english_translated"] = translate_forward_checkpointed(
        translator, df,
        src=p["source_lang"], tgt=p["target_lang"], batch_size=p["batch_size"],
        ckpt_path=ckpt_path,
    ).values
    df["translation_tool"] = model_name
    df["repetition_flag"] = df["english_translated"].map(is_degenerate)
    n_deg = int(df["repetition_flag"].sum())
    if n_deg:
        print(f"[stage-1] WARNING: {n_deg} translations flagged as repetition-degenerate")

    # ---- quality assurance via back-translation ----
    qa = back_translation_quality(
        translator, df, p["sample_size_for_validation"],
        src=p["source_lang"], tgt=p["target_lang"], batch_size=p["batch_size"],
        bertscore_model=p["bertscore_model"], seed=p["seed"],
        baseline_pairs=p["baseline_pairs"],
    )
    raw = qa["raw_scores"]
    rescaled = qa["rescaled_scores"]
    baseline = qa["baseline"]

    # quality_score column carries the (interpretable) rescaled score
    df["quality_score"] = pd.NA
    df["quality_score_raw"] = pd.NA
    for i, rs, rw in zip(qa["sample_idx"], rescaled, raw):
        df.at[i, "quality_score"] = round(rs, 4)
        df.at[i, "quality_score_raw"] = round(rw, 4)

    thr = p["quality_threshold"]           # applied to rescaled scores
    n_pass = sum(s >= thr for s in rescaled)
    mean_rescaled = sum(rescaled) / len(rescaled) if rescaled else 0.0
    mean_raw = sum(raw) / len(raw) if raw else 0.0
    problematic = [
        {"review_id": df.at[i, "review_id"],
         "quality_score": round(rs, 4), "quality_score_raw": round(rw, 4)}
        for i, rs, rw in zip(qa["sample_idx"], rescaled, raw) if rs < thr
    ]

    # ---- add English facility-type label ----
    if "facility_type" in df.columns:
        from config import FACILITY_TYPE_EN
        df["facility_type_en"] = (df["facility_type"].map(FACILITY_TYPE_EN)
                                  .fillna(df["facility_type"]))

    # ---- write outputs ----
    # metadata first (facility_type sits beside the content); then translations
    meta_cols = [c for c in ("review_id", "review_id_src", "sentiment", "source_url",
                             "title", "facility_type", "facility_type_en")
                 if c in df.columns]
    derived = ["original_russian", "english_translated", "translation_tool",
               "repetition_flag", "quality_score", "quality_score_raw"]
    out_cols = meta_cols + [c for c in derived if c not in meta_cols]
    csv_path = OUTPUT / f"translations_{args.tag}.csv"
    df[out_cols].to_csv(csv_path, index=False, encoding="utf-8")

    report = {
        "dataset": p["dataset_name"],
        "model": p["model_name"],
        "device": device,
        "num_total": int(len(df)),
        "num_validated": int(len(rescaled)),
        "scoring": "back-translation BERTScore, empirically baseline-rescaled",
        "bertscore_model": p["bertscore_model"],
        "empirical_baseline": round(baseline, 4),
        "num_pass_threshold": int(n_pass),
        "pct_pass_threshold": round(100 * n_pass / len(rescaled), 2) if rescaled else 0.0,
        "mean_score_rescaled": round(mean_rescaled, 4),
        "mean_score_raw": round(mean_raw, 4),
        "quality_threshold": thr,
        "problematic_reviews": problematic,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    report_path = REPORTS / f"translation_quality_{args.tag}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[stage-1] wrote {csv_path}")
    print(f"[stage-1] wrote {report_path}")
    print(f"[stage-1] empirical ru baseline = {baseline:.4f}")
    print(f"[stage-1] QA: {n_pass}/{len(rescaled)} >= {thr} rescaled "
          f"(mean_rescaled={mean_rescaled:.4f}, mean_raw={mean_raw:.4f}) "
          f"in {report['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
