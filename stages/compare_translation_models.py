"""
Benchmark free/local Russian->English translation models on the same sample,
using the same QA metric as Stage -1: back-translation (ru->en->ru) BERTScore,
empirically baseline-rescaled so random~0 and perfect~1.

Models compared (all free, run locally, no API):
  - m2m100 : facebook/m2m100_418M          (current Stage -1 baseline)
  - opus   : Helsinki-NLP/opus-mt-ru-en + opus-mt-en-ru  (ru<->en specialists)
  - nllb   : facebook/nllb-200-distilled-600M

Quick run (downloads ~3 GB of models on first use):
    python stages/compare_translation_models.py --sample 10

Outputs:
    data/output/translation_model_comparison.csv   (per-review, every model)
    reports/translation_model_comparison.json       (summary scores + timing)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import STAGE_M1, OUTPUT, REPORTS  # noqa: E402
from stages.stage_m1_translate import (  # noqa: E402
    load_reviews, _bertscore_f1, resolve_device,
)

RU, EN = "ru", "en"


# ----------------------------------------------------------------------------
# Model adapters — common interface: .translate(texts, src, tgt) -> list[str]
# ----------------------------------------------------------------------------
class M2M100Model:
    key, name = "m2m100", "facebook/m2m100_418M"

    def __init__(self, device, max_len):
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        self.tok = M2M100Tokenizer.from_pretrained(self.name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(self.name).to(device)
        self.model.eval()
        self.device, self.max_len = device, max_len

    def translate(self, texts, src, tgt, bs=8):
        import torch
        self.tok.src_lang = src
        forced = self.tok.get_lang_id(tgt)
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], return_tensors="pt", padding=True,
                           truncation=True, max_length=self.max_len).to(self.device)
            with torch.no_grad():
                g = self.model.generate(**enc, forced_bos_token_id=forced,
                                        max_length=self.max_len)
            out.extend(self.tok.batch_decode(g, skip_special_tokens=True))
        return out


class OpusMTModel:
    key, name = "opus", "Helsinki-NLP/opus-mt-{pair}"

    def __init__(self, device, max_len):
        from transformers import MarianMTModel, MarianTokenizer
        self.device, self.max_len = device, max_len
        self.mods = {}
        for pair in ("ru-en", "en-ru"):
            n = f"Helsinki-NLP/opus-mt-{pair}"
            tok = MarianTokenizer.from_pretrained(n)
            m = MarianMTModel.from_pretrained(n).to(device)
            m.eval()
            self.mods[pair] = (tok, m)

    def translate(self, texts, src, tgt, bs=8):
        import torch
        tok, m = self.mods[f"{src}-{tgt}"]
        out = []
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                      truncation=True, max_length=self.max_len).to(self.device)
            with torch.no_grad():
                g = m.generate(**enc, max_length=self.max_len)
            out.extend(tok.batch_decode(g, skip_special_tokens=True))
        return out


class NLLBModel:
    key, name = "nllb", "facebook/nllb-200-distilled-600M"
    codes = {"ru": "rus_Cyrl", "en": "eng_Latn"}

    def __init__(self, device, max_len):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.name).to(device)
        self.model.eval()
        self.device, self.max_len = device, max_len

    def translate(self, texts, src, tgt, bs=8):
        import torch
        self.tok.src_lang = self.codes[src]
        forced = self.tok.convert_tokens_to_ids(self.codes[tgt])
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], return_tensors="pt", padding=True,
                           truncation=True, max_length=self.max_len).to(self.device)
            with torch.no_grad():
                g = self.model.generate(**enc, forced_bos_token_id=forced,
                                        max_length=self.max_len)
            out.extend(self.tok.batch_decode(g, skip_special_tokens=True))
        return out


class OpusTCBigModel:
    """OPUS-MT tc-big (MarianMT). ru->en via East-Slavic->En (many-to-one);
    en->ru via En->East-Slavic, which needs a '>>rus<<' target-language tag."""
    key, name = "opus_tc_big", "Helsinki-NLP/opus-mt-tc-big-{zle-en,en-zle}"

    def __init__(self, device, max_len):
        from transformers import MarianMTModel, MarianTokenizer
        self.device, self.max_len = device, max_len
        self.mods = {}
        for pair in ("zle-en", "en-zle"):
            n = f"Helsinki-NLP/opus-mt-tc-big-{pair}"
            tok = MarianTokenizer.from_pretrained(n)
            m = MarianMTModel.from_pretrained(n).to(device)
            m.eval()
            self.mods[pair] = (tok, m)

    def translate(self, texts, src, tgt, bs=8):
        import torch
        if (src, tgt) == ("ru", "en"):
            tok, m = self.mods["zle-en"]
            prep = list(texts)
        elif (src, tgt) == ("en", "ru"):
            tok, m = self.mods["en-zle"]
            prep = [">>rus<< " + t for t in texts]   # target-language tag required
        else:
            raise ValueError(f"unsupported direction {src}->{tgt}")
        out = []
        for i in range(0, len(prep), bs):
            enc = tok(prep[i:i + bs], return_tensors="pt", padding=True,
                      truncation=True, max_length=self.max_len).to(self.device)
            with torch.no_grad():
                g = m.generate(**enc, max_length=self.max_len)
            out.extend(tok.batch_decode(g, skip_special_tokens=True))
        return out


class NLLB13BModel(NLLBModel):
    """Bigger NLLB — inherits translate(); only the checkpoint name changes."""
    key, name = "nllb_1.3B", "facebook/nllb-200-distilled-1.3B"


class LlamaCppModel:
    """Prompt-based LLM translation via a 4-bit GGUF Llama-2-7B-Chat on CPU."""
    key, name = "llama2", "TheBloke/Llama-2-7B-Chat-GGUF Q4_K_M"
    repo, fname = "TheBloke/Llama-2-7B-Chat-GGUF", "llama-2-7b-chat.Q4_K_M.gguf"
    LANG = {"ru": "Russian", "en": "English"}

    def __init__(self, device, max_len):
        import os
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(self.repo, self.fname)
        self.llm = Llama(model_path=path, n_ctx=4096,
                         n_threads=max(1, (os.cpu_count() or 4) - 1), verbose=False)
        self.max_len = max_len

    @staticmethod
    def _clean(text: str) -> str:
        # strip common chat preambles ("Sure! Here is the translation:")
        t = text.strip().strip('"')
        for marker in ("\n\n", ":\n", ": "):
            low = t.lower()
            if low.startswith(("sure", "here", "of course", "certainly")) and marker in t:
                t = t.split(marker, 1)[1].strip().strip('"')
                break
        return t

    def translate(self, texts, src, tgt, bs=8):
        out = []
        n = len(texts)
        for i, t in enumerate(texts):
            prompt = (
                "<s>[INST] <<SYS>>\nYou are a professional translator.\n<</SYS>>\n\n"
                f"Translate the following {self.LANG[src]} text to {self.LANG[tgt]}. "
                f"Output ONLY the {self.LANG[tgt]} translation, no preamble or notes.\n\n"
                f"{t} [/INST]"
            )
            r = self.llm(prompt, max_tokens=512, temperature=0.1,
                         stop=["</s>", "[INST]"])
            out.append(self._clean(r["choices"][0]["text"]))
            print(f"[compare]   llama {i + 1}/{n}", end="\r", flush=True)
        print()
        return out


class GGUFChatModel:
    """Generic prompt-based translator over a GGUF chat model (llama-cpp-python).
    Subclasses set repo + build_prompt(). Q4_K_M file is auto-detected."""
    key = name = repo = None
    n_ctx = 4096
    stop = ["</s>"]
    LANG = {"ru": "Russian", "en": "English"}

    def __init__(self, device, max_len):
        import os
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(self.repo, self._find_gguf())
        self.llm = Llama(model_path=path, n_ctx=self.n_ctx,
                         n_threads=max(1, (os.cpu_count() or 4) - 1), verbose=False)
        self.max_len = max_len

    @classmethod
    def _find_gguf(cls):
        from huggingface_hub import HfApi
        q4 = [f for f in HfApi().list_repo_files(cls.repo)
              if "Q4_K_M" in f and f.endswith(".gguf")]
        if not q4:
            raise FileNotFoundError(f"no Q4_K_M gguf in {cls.repo}")
        return q4[0]

    def build_prompt(self, text, src, tgt):
        raise NotImplementedError

    @staticmethod
    def clean(text):
        t = str(text).strip().strip('"')
        low = t.lower()
        if low.startswith(("sure", "here", "of course", "certainly", "translation")):
            for mk in (":\n", ": ", "\n\n"):
                if mk in t:
                    t = t.split(mk, 1)[1].strip().strip('"')
                    break
        return t

    def translate(self, texts, src, tgt, bs=8):
        out, n = [], len(texts)
        for i, t in enumerate(texts):
            r = self.llm(self.build_prompt(t, src, tgt), max_tokens=512,
                         temperature=0.1, stop=self.stop)
            out.append(self.clean(r["choices"][0]["text"]))
            print(f"[compare]   {self.key} {i + 1}/{n}", end="\r", flush=True)
        print()
        return out


class MistralNemoModel(GGUFChatModel):
    key, name = "mistral_nemo", "Mistral-Nemo-Instruct-2407 Q4_K_M"
    repo = "bartowski/Mistral-Nemo-Instruct-2407-GGUF"
    stop = ["</s>", "[INST]"]

    def build_prompt(self, text, src, tgt):
        return (f"[INST] Translate the following {self.LANG[src]} text to "
                f"{self.LANG[tgt]}. Output ONLY the {self.LANG[tgt]} translation, "
                f"with no preamble or notes.\n\n{text}[/INST]")


REGISTRY = {
    "m2m100": M2M100Model, "opus": OpusMTModel, "nllb": NLLBModel,
    "opus_tc_big": OpusTCBigModel, "nllb_1.3B": NLLB13BModel,
    "llama2": LlamaCppModel, "mistral_nemo": MistralNemoModel,
}


# ----------------------------------------------------------------------------
def empirical_baseline(originals: list[str], n_pairs: int, bertscore_model: str,
                       seed: int) -> float:
    """Mean BERTScore of random unrelated ru pairs (the 'chance' floor)."""
    random.seed(seed)
    pool = random.sample(originals, min(n_pairs, len(originals)))
    shifted = pool[1:] + pool[:1]
    scores = _bertscore_f1(pool, shifted, bertscore_model)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_model(ModelCls, sample: pd.DataFrame, baseline: float,
                   device: str, max_len: int, bertscore_model: str) -> dict:
    t0 = time.time()
    print(f"\n[compare] === {ModelCls.key} ({ModelCls.name}) ===")
    model = ModelCls(device, max_len)
    originals = sample["original_russian"].tolist()

    print(f"[compare]   translating {len(originals)} ru->en ...")
    en = model.translate(originals, RU, EN)
    print(f"[compare]   back-translating en->ru ...")
    back = model.translate(en, EN, RU)

    raw = _bertscore_f1(back, originals, bertscore_model)
    denom = (1.0 - baseline) or 1e-9
    rescaled = [(r - baseline) / denom for r in raw]
    elapsed = time.time() - t0
    mean_rs = sum(rescaled) / len(rescaled)
    print(f"[compare]   mean_rescaled={mean_rs:.4f} "
          f"mean_raw={sum(raw)/len(raw):.4f} time={elapsed:.0f}s")
    return {
        "key": ModelCls.key, "name": ModelCls.name,
        "english": en, "raw": raw, "rescaled": rescaled,
        "mean_rescaled": mean_rs,
        "median_rescaled": sorted(rescaled)[len(rescaled) // 2],
        "mean_raw": sum(raw) / len(raw),
        "elapsed_seconds": round(elapsed, 1),
        "sec_per_review": round(elapsed / max(1, len(originals)), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=10, help="reviews to test")
    ap.add_argument("--models", type=str, default="m2m100,opus,nllb",
                    help="comma-separated subset of: " + ",".join(REGISTRY))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--append", action="store_true",
                    help="reuse existing comparison CSV; only run models not already in it")
    args = ap.parse_args()

    device = resolve_device(STAGE_M1["device"])
    max_len = STAGE_M1["max_input_tokens"]
    bsm = STAGE_M1["bertscore_model"]

    full = load_reviews(None)
    sample = full.sample(args.sample, random_state=args.seed).reset_index(drop=True)
    print(f"[compare] testing {len(sample)} reviews on device={device}")

    baseline = empirical_baseline(full["original_russian"].tolist(),
                                  STAGE_M1["baseline_pairs"], bsm, args.seed)
    print(f"[compare] empirical ru baseline = {baseline:.4f}")

    csv_path = OUTPUT / "translation_model_comparison.csv"
    json_path = REPORTS / "translation_model_comparison.json"

    # ---- when appending, load prior results and reuse models already present ----
    out = sample[["review_id", "original_russian"]].copy()
    prior_entries: list[dict] = []
    if args.append and csv_path.exists():
        prev = pd.read_csv(csv_path)
        if len(prev) == len(out) and (prev["review_id"].values == out["review_id"].values).all():
            for col in prev.columns:
                if col.endswith("_en") or col.endswith("_score"):
                    out[col] = prev[col].values
            if json_path.exists():
                prior_entries = json.loads(json_path.read_text(encoding="utf-8")).get("models", [])
            print(f"[compare] append: reusing columns {[c for c in prev.columns if c.endswith('_en')]}")
        else:
            print("[compare] append: existing CSV sample mismatch -> ignoring it")

    chosen = [m.strip() for m in args.models.split(",") if m.strip()]
    results = []
    for key in chosen:
        if key not in REGISTRY:
            print(f"[compare] skip unknown model '{key}'")
            continue
        if args.append and f"{key}_en" in out.columns:
            print(f"[compare] reuse cached '{key}' (already in CSV)")
            continue
        try:
            results.append(evaluate_model(REGISTRY[key], sample, baseline,
                                          device, max_len, bsm))
        except Exception as e:  # keep going if one model fails
            print(f"[compare] MODEL '{key}' FAILED: {type(e).__name__}: {e}")

    # ---- add freshly-run models to the per-review CSV ----
    for r in results:
        out[f"{r['key']}_en"] = r["english"]
        out[f"{r['key']}_score"] = [round(x, 4) for x in r["rescaled"]]
    out.to_csv(csv_path, index=False, encoding="utf-8")

    # ---- summary JSON (merge fresh entries with reused prior ones) ----
    fresh = [{"key": r["key"], "name": r["name"],
              "mean_rescaled": round(r["mean_rescaled"], 4),
              "median_rescaled": round(r["median_rescaled"], 4),
              "mean_raw": round(r["mean_raw"], 4),
              "elapsed_seconds": r["elapsed_seconds"],
              "sec_per_review": r["sec_per_review"]} for r in results]
    fresh_keys = {e["key"] for e in fresh}
    kept = [e for e in prior_entries if e["key"] in
            {c[:-3] for c in out.columns if c.endswith("_en")} and e["key"] not in fresh_keys]
    summary = {
        "num_reviews": len(sample), "device": device,
        "bertscore_model": bsm, "empirical_baseline": round(baseline, 4),
        "seed": args.seed,
        "models": sorted(fresh + kept, key=lambda d: d["mean_rescaled"], reverse=True),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    # ---- console leaderboard ----
    print("\n[compare] ===== LEADERBOARD (higher rescaled = better) =====")
    print(f"{'model':<26}{'mean':>8}{'median':>8}{'raw':>8}{'s/rev':>8}")
    for m in summary["models"]:
        print(f"{m['key']:<26}{m['mean_rescaled']:>8.3f}"
              f"{m['median_rescaled']:>8.3f}{m['mean_raw']:>8.3f}"
              f"{m['sec_per_review']:>8.1f}")
    print(f"\n[compare] wrote {csv_path}")
    print(f"[compare] wrote {json_path}")


if __name__ == "__main__":
    main()
