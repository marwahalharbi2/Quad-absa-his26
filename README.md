# Modular Quad-ABSA Pipeline for Health-Oriented Hospitality Reviews

Code, data, and the annotation protocol for the paper:

> **Large Language Models for Quadruple Aspect-Based Sentiment Analysis in Health-Oriented Hospitality Reviews**
> Marwah Alharbi, Majid Alharbi, Jiao Yin, Lili Sun — HIS 2026.
> Springer LNCS 17294, https://doi.org/10.1007/978-981-92-7708-7_12

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23680928.svg)](https://doi.org/10.5281/zenodo.23680928)
Archived on Zenodo: https://doi.org/10.5281/zenodo.23680928

A modular pipeline that turns raw, multilingual healthcare-facility reviews into
fine-grained, decision-relevant insight via **Quadruple Aspect-Based Sentiment
Analysis (Quad-ABSA)** — decomposing each review into *(aspect term, aspect
category, opinion term, sentiment)*. Three off-the-shelf LLMs
(**Mistral-Nemo-12B**, **GPT-5.5**, **Claude**) are benchmarked against a human
gold standard under both exact-match F1 and **SAFE** (Semantic-Aware Flexible
Evaluation). No fine-tuning.

## Headline result

Exact-match F1 drastically under-reports extraction quality; SAFE reveals a
~4–8× higher score. On the 50-review human gold (95% CIs from a review-level
cluster bootstrap, B=5,000):

| Model | Exact-F1 | SAFE-F1 [95% CI] | mean-SAFE | Precision | Recall |
|-------|:--------:|:----------------:|:---------:|:---------:|:------:|
| **Mistral-Nemo-12B** | 13.8 | **75.9** [71–80] | 71.9 | 75 | 76 |
| Claude | 8.5 | 68.9 [63–75] | 84.0 | 89 | 56 |
| GPT-5.5 | 19.5 | 68.4 [61–75] | 62.0 | 57 | 86 |

Mistral-Nemo is the balanced winner (robust across bootstrap resamples and the
truncation split) and is used for full-corpus extraction: 3,951 quadruples over
1,000 reviews. Full analysis is in the paper.

## Pipeline

Six interchangeable stages, numbered −1 through 4. Stages −1 and 0 are optional,
dataset-dependent preprocessing; Stages 1–4 are the core pipeline. Validation
(Stage 2) deliberately precedes full extraction (Stage 3).

| Stage | What | Scripts | Key output |
|-------|------|---------|------------|
| **−1** Translate (ru→en) + QA | OPUS-MT-tc-big; round-trip BERTScore QA; long-review repair | `stages/stage_m1_translate.py`, `stages/compare_translation_models.py`, `stages/hybrid_fix_long_reviews.py` | `data/output/translations_opus1000_hybrid.csv` |
| **0** Aspect discovery (LDA) | LDA (K=30) → 6-category taxonomy + *Other* | `analysis/lda_topics.py` | `data/output/lda_topics_30.csv`, `lda_aspect_themes.csv` |
| **1** Annotation | Human gold + 3 LLM annotators on a stratified 100-review sample; 50-review complete-case eval set | `stages/make_mistral_100.py`, `stages/clean_annotation.py`, `stages/select_eval_50.py` | `data/output/*_100_annotation.csv`, `human_annotation_template_50.xlsx`, `eval_50_ids.csv` |
| **2** Model comparison | Exact-F1 + SAFE (MiniLM `all-MiniLM-L6-v2`, τ=0.70); bootstrap CIs; truncation robustness | `stages/eval_human_partial.py`, `stages/bootstrap_ci.py`, `stages/eval_truncation_split.py`, `stages/safe_eval.py` | `reports/` (generated on run) |
| **3** Full-corpus extraction | Winning model over all 1,000 reviews | `stages/stage2_llm_extract.py`, `stages/assemble_final_absa.py` | `data/output/Mistral_predicted_quadruples_final.csv` (3,951 quads) |
| **4** Analysis & insight | Category × polarity aggregation | (see paper) | — |

The written annotation protocol — a domain-general contribution — is
[`docs/annotation_guidelines.md`](docs/annotation_guidelines.md).
The taxonomy derivation (LDA topics → themes → categories) is in
[`docs/taxonomy_derivation.md`](docs/taxonomy_derivation.md).

## Repository layout

```
stages/        pipeline scripts (translate, annotate-clean, extract, evaluate)
analysis/      LDA aspect discovery
config.py      all tunable parameters
data/output/   translations, annotations, extracted quadruples, eval set
docs/          annotation guidelines (the released protocol), taxonomy derivation
```

Evaluation outputs (`reports/`) are **generated** by running the Stage 2 scripts;
they are not checked in.

## Environment

Anaconda **base** (Python 3.11), **CPU-only**. Key pins: `torch>=2.6`,
`llama-cpp-python==0.2.90`, `sentence-transformers`.

```bash
python -m pip install -r requirements.txt
```

## Reproducing the evaluation

```bash
# SAFE + exact-F1 vs the human gold (Stage 2)
python stages/eval_human_partial.py

# bootstrap 95% CIs over the 50 reviews
python stages/bootstrap_ci.py

# truncation robustness (clean-38 vs truncated-12)
python stages/eval_truncation_split.py
```

All parameters live in `config.py`. Long runs are checkpointed and resumable.

## Data & ethics

The corpus derives from the public
[`blinoff/healthcare_facilities_reviews`](https://huggingface.co/datasets/blinoff/healthcare_facilities_reviews)
dataset (70,597 Russian-language reviews). This repository ships a
sentiment-stratified sample (1,000 reviews + a 200 top-up pool), its English
translations, and derived annotations/quadruples, for research reproducibility.
These are **user-generated reviews** that may mention individuals; rights to the
underlying text remain with the original authors and the source dataset (no
separate license is asserted over the review text). Please use responsibly and
cite the source dataset below.
Source review URLs are not included.

## License

CC BY 4.0 (see [`LICENSE`](LICENSE)) for the code, translations and annotations.
The review text is covered by the data note above.

## Citation

```bibtex
@inproceedings{alharbi2026quadabsa,
  author    = {Alharbi, Marwah and Alharbi, Majid and Yin, Jiao and Sun, Lili},
  title     = {Large Language Models for Quadruple Aspect-Based Sentiment
               Analysis in Health-Oriented Hospitality Reviews},
  booktitle = {Health Information Science (HIS 2026)},
  series    = {Lecture Notes in Computer Science},
  volume    = {17294},
  publisher = {Springer},
  year      = {2027},
  doi       = {10.1007/978-981-92-7708-7_12}
}
```

Source dataset: Blinov, P. *Dataset of Russian reviews about medical facilities*
(2022), Hugging Face.
