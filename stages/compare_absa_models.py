"""
Compare free/local LLMs on Quad-ABSA extraction over the same reviews.

For each English review, each model extracts quadruples:
  (aspect_category, aspect_term, opinion_term, sentiment)
- aspect_category: one of config.ASPECT_CATEGORIES (6 + "Other")
- aspect_term:  the phrase from the text that gives away the aspect (a noun OR an
                action phrase, e.g. "be admitted"); "IMPLICIT" only if nothing does
- opinion_term: the phrase expressing the evaluation; "IMPLICIT" if none
- sentiment:    positive | negative | neutral

Per-review checkpointed (resumable). Runs on CPU via llama-cpp-python.

    python stages/compare_absa_models.py --sample 10 --models llama2,mistral_nemo

Outputs:
  data/output/absa_comparison.csv    (long format: one row per extracted quad)
  reports/absa_comparison.json        (per-model summary: counts, categories, time)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ASPECT_CATEGORIES, OUTPUT, REPORTS  # noqa: E402

SRC = OUTPUT / "absa_input.csv"
OUT_CSV = OUTPUT / "absa_comparison.csv"
CKPT = OUTPUT.parent / "interim" / "absa_comparison_partial.csv"
SUMMARY = REPORTS / "absa_comparison.json"

N_CTX = 8192
MAX_TOKENS = 1024
VALID_SENTIMENT = {"positive", "negative", "neutral"}

# ---- GGUF model specs (all cached from the translation benchmark) ----
MODELS = {
    "llama2": {
        "name": "Llama-2-7B-Chat Q4_K_M",
        "repo": "TheBloke/Llama-2-7B-Chat-GGUF",
        "file": "llama-2-7b-chat.Q4_K_M.gguf",
        "fmt": "llama2",
    },
    "mistral_nemo": {
        "name": "Mistral-Nemo-Instruct-2407 Q4_K_M",
        "repo": "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
        "file": "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
        "fmt": "mistral",
    },
    "claude": {
        "name": "Claude Opus 4.8",
        "kind": "api",
        "model": "claude-opus-4-8",
    },
}


def json_schema() -> dict:
    """Structured-output schema: guarantees valid quadruples from the API model."""
    return {
        "type": "object",
        "properties": {
            "quadruples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "aspect_category": {"type": "string",
                                            "enum": list(ASPECT_CATEGORIES.keys())},
                        "aspect_term": {"type": "string"},
                        "opinion_term": {"type": "string"},
                        "sentiment": {"type": "string",
                                      "enum": ["positive", "negative", "neutral"]},
                    },
                    "required": ["aspect_category", "aspect_term",
                                 "opinion_term", "sentiment"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["quadruples"],
        "additionalProperties": False,
    }


def build_system() -> str:
    cats = "\n".join(f"{i}. {k} — {v}" for i, (k, v) in enumerate(ASPECT_CATEGORIES.items(), 1))
    return (
        "You are an expert annotator for Aspect-Based Sentiment Analysis (ABSA) "
        "of healthcare-facility reviews.\n\n"
        "Extract EVERY aspect the reviewer expresses an opinion about. A review "
        "may contain several aspects; output one quadruple per aspect. Each "
        "quadruple has:\n"
        "- aspect_category: EXACTLY one category name from the list below.\n"
        "- aspect_term: the shortest phrase COPIED from the review that gives "
        "away the aspect. It can be a noun (\"nurses\", \"equipment\") OR an "
        "action phrase (\"be admitted\", \"waiting for results\"). Use "
        "\"IMPLICIT\" only if truly nothing in the text reveals the aspect.\n"
        "- opinion_term: the shortest phrase COPIED from the review that "
        "expresses the evaluation (\"takes ages\", \"very professional\"). Use "
        "\"IMPLICIT\" if there is none.\n"
        "- sentiment: positive, negative, or neutral.\n\n"
        "COUNTING RULE - the unit is the (aspect, opinion) PAIR, exactly as a human "
        "annotator would count:\n"
        "- Two different aspects that SHARE one opinion => TWO quadruples (repeat the "
        "opinion in each). \"the doctors and nurses were rude\" => one quad for "
        "doctors, one for nurses.\n"
        "- One aspect with TWO opinions => TWO quadruples (repeat the aspect in each). "
        "\"service is cheap but slow\" => one quad for cheap, one for slow.\n\n"
        "SCOPE RULE - annotate ONLY the facility being reviewed. If the review mentions "
        "or compares against a DIFFERENT named facility (or a previous doctor/clinic used "
        "only as a contrast), do NOT extract opinions about that other facility.\n\n"
        f"Categories:\n{cats}\n\n"
        "Example:\n"
        "Review: \"It takes ages to be admitted to the ER, but the doctors were kind.\"\n"
        "Output: {\"quadruples\": ["
        "{\"aspect_category\": \"Operational Experience\", \"aspect_term\": \"be admitted\", "
        "\"opinion_term\": \"takes ages\", \"sentiment\": \"negative\"}, "
        "{\"aspect_category\": \"Staff & Service\", \"aspect_term\": \"doctors\", "
        "\"opinion_term\": \"kind\", \"sentiment\": \"positive\"}]}\n\n"
        "Return ONLY a JSON object of the form {\"quadruples\": [ ... ]}. "
        "No explanations, no extra text."
    )


def build_prompt(fmt: str, system: str, review: str) -> str:
    user = f"Review:\n{review}"
    if fmt == "llama2":
        return f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] "
    # mistral has no system role; prepend the instructions to the user turn
    return f"[INST] {system}\n\n{user}[/INST] "


def parse_quads(text: str) -> tuple[list[dict], bool]:
    """Tolerantly pull the quadruple list out of a model's raw output."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    obj = None
    # try whole string, then the first {...} or [...] blob
    for candidate in (t,):
        try:
            obj = json.loads(candidate)
            break
        except Exception:
            pass
    if obj is None:
        m = re.search(r"\{.*\}", t, re.DOTALL) or re.search(r"\[.*\]", t, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                return [], False
    if obj is None:
        return [], False
    quads = obj.get("quadruples", obj) if isinstance(obj, dict) else obj
    if not isinstance(quads, list):
        return [], False
    out = []
    for q in quads:
        if not isinstance(q, dict):
            continue
        out.append({
            "aspect_category": str(q.get("aspect_category", "")).strip(),
            "aspect_term": str(q.get("aspect_term", "")).strip(),
            "opinion_term": str(q.get("opinion_term", "")).strip(),
            "sentiment": str(q.get("sentiment", "")).strip().lower(),
        })
    return out, True


def build_extractor(spec: dict, system: str):
    """Return a function review_text -> raw model output (JSON text)."""
    if spec.get("kind") == "api":
        import anthropic
        client = anthropic.Anthropic()
        schema = json_schema()

        def extract(review: str) -> str:
            resp = client.messages.create(
                model=spec["model"], max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": f"Review:\n{review}"}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
            return next(b.text for b in resp.content if b.type == "text")
        return extract

    # local GGUF model
    import os
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(spec["repo"], spec["file"])
    llm = Llama(model_path=path, n_ctx=N_CTX,
                n_threads=max(1, (os.cpu_count() or 4) - 1), verbose=False)

    def extract(review: str) -> str:
        out = llm(build_prompt(spec["fmt"], system, review),
                  max_tokens=MAX_TOKENS, temperature=0.1, stop=["</s>", "[INST]"])
        return out["choices"][0]["text"]
    return extract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--models", type=str, default="llama2,mistral_nemo")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(SRC).sample(args.sample, random_state=args.seed).reset_index(drop=True)
    system = build_system()
    keys = [m.strip() for m in args.models.split(",") if m.strip()]

    rows = []
    done = set()               # (model_key, review_id) already processed
    if CKPT.exists():
        prev = pd.read_csv(CKPT)
        rows = prev.to_dict("records")
        done = {(r["model"], r["review_id"]) for r in rows}
        print(f"[absa] resume: {len(done)} (model,review) pairs done")

    # preserve summaries for models already computed in a previous run
    summary = {}
    if SUMMARY.exists():
        try:
            summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        except Exception:
            summary = {}

    for key in keys:
        spec = MODELS[key]
        t0 = time.time()
        print(f"\n[absa] === {key} ({spec['name']}) ===")
        extract = build_extractor(spec, system)
        n_quads = n_err = 0
        for i, r in df.iterrows():
            rid = r["review_id"]
            if (key, rid) in done:
                continue
            quads, ok = parse_quads(extract(str(r["english_translated"])))
            if not ok:
                n_err += 1
            if not quads:
                rows.append({"model": key, "review_id": rid,
                             "english_translated": r["english_translated"],
                             "aspect_category": "", "aspect_term": "",
                             "opinion_term": "", "sentiment": "", "parse_ok": ok})
            for q in quads:
                n_quads += 1
                rows.append({"model": key, "review_id": rid,
                             "english_translated": r["english_translated"], **q,
                             "parse_ok": ok})
            pd.DataFrame(rows).to_csv(CKPT, index=False, encoding="utf-8")  # checkpoint
            print(f"[absa]   {key} {i + 1}/{len(df)} {rid}: {len(quads)} quads", flush=True)
        elapsed = time.time() - t0
        mrows = [r for r in rows if r["model"] == key and r.get("aspect_category")]
        cats = pd.Series([r["aspect_category"] for r in mrows]).value_counts().to_dict()
        summary[key] = {
            "name": spec["name"], "num_reviews": len(df),
            "num_quads": len(mrows), "parse_errors": n_err,
            "avg_quads_per_review": round(len(mrows) / len(df), 2),
            "category_distribution": cats,
            "elapsed_seconds": round(elapsed, 1),
        }
        print(f"[absa]   {key}: {len(mrows)} quads, {n_err} parse errors, "
              f"{elapsed:.0f}s")

    final = pd.DataFrame(rows)
    order = ["review_id", "model", "aspect_category", "aspect_term",
             "opinion_term", "sentiment", "parse_ok", "english_translated"]
    final = final[[c for c in order if c in final.columns]]
    final.to_csv(OUT_CSV, index=False, encoding="utf-8")
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[absa] wrote {OUT_CSV}")
    print(f"[absa] wrote {SUMMARY}")
    for k, s in summary.items():
        print(f"  {k}: {s['num_quads']} quads, avg {s['avg_quads_per_review']}/review, "
              f"{s['parse_errors']} parse errors, {s['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
