# `data/output/` — Data Dictionary

Outputs of the Paper 6 Quad-ABSA pipeline (HIS 2026). Files are grouped by
pipeline stage. All reviews share a stable `review_id` (`rNNNNNN`) so files can be
joined across stages. Source dataset: `blinoff/healthcare_facilities_reviews`
(Russian healthcare-facility reviews).

Pipeline: **Stage -1** translate → **Stage 0/0b** LDA + taxonomy → **Stage 2**
LLM quad extraction → **Stage 3** (F1 + SAFE) evaluation *(not built yet)*.

---

## Stage -1 — Translation (Russian → English)

### `translations_opus1000_hybrid.csv`
The clean translated corpus: **1,000** reviews (999 clean, 1 still flagged).
Hybrid MT: 941 via OPUS-MT-tc-big + 59 long reviews re-translated with Mistral-Nemo.
**Columns:** `review_id, review_id_src, sentiment, source_url, title, facility_type,
facility_type_en, original_russian, english_translated, translation_tool,
repetition_flag, quality_score, quality_score_raw, en_ru_len_ratio, truncated_flag`.
`en_ru_len_ratio` = English/Russian char-length ratio; `truncated_flag` = ratio < 0.65
(marks long reviews silently truncated by OPUS-MT's ~512-token cap — 239/1000; see the
draft's "Translation truncation" limitation; re-translation deferred post-submission).
*Purpose:* the base translated dataset for everything downstream.

### `translations_opus1200.csv`
Same schema, **extended to 1,200** reviews (the original 1,000 + a 200-review
top-up). *Purpose:* source of the fresh reviews used to replace the 140 that later
failed extraction.

### `translation_model_comparison.csv`
Benchmark of **7 MT models** on a common sample, scored by round-trip
(back-translation) rescaled BERTScore. **Columns:** `review_id, original_russian`,
then `<model>_en` + `<model>_score` for m2m100, opus, nllb, nllb_1.3B,
**opus_tc_big**, llama2, mistral_nemo. *Purpose:* model selection — **OPUS-MT-tc-big
chosen** (best quality/speed).

---

## Stage 0 / 0b — Aspect discovery & taxonomy

### `lda_topics_30.csv`
LDA (K=30) over the English reviews. **Columns:** `topic_id` (T1–T30),
`coherence`, `top_words`, `theme`. *Purpose:* data-driven aspect discovery.

### `lda_aspect_themes.csv`
Human consolidation of the 30 topics into aspect themes. **Columns:**
`aspect_theme, lda_topics`. *Purpose:* the trail from LDA topics → the final
**6-category + Other** taxonomy (see `config.ASPECT_CATEGORIES`).

---

## Stage 2 — LLM Quad-ABSA extraction

### `absa_input.csv`
Extractor input for the 1,000 reviews. **Columns:** `review_id, content`
(original Russian), `facility_type_en, english_translated`. *Purpose:* what the
LLM reads (it extracts from `english_translated`).

### `absa_input_new200.csv`
Same schema for the 200-review top-up batch.

### `absa_comparison.csv`
Free-model bake-off (Llama-2-7B vs Mistral-Nemo) on 10 reviews, long format
(one row per quad). **Columns:** `review_id, model, aspect_category, aspect_term,
opinion_term, sentiment, parse_ok, english_translated`. *Purpose:* extraction-model
selection — **Mistral-Nemo chosen** (Llama-2 unreliable).

### `Mistral_predicted_quadruples.csv`  *(intermediate)*
Mistral-Nemo's **first pass** over the original 1,000. **3,420 quads over 860
reviews; 140 reviews parse-failed** (empty rows, marked by `parse_ok=False`).
**Columns:** `review_id, facility_type_en, aspect_category, aspect_term,
opinion_term, sentiment, parse_ok, english_translated`.

### `Mistral_predicted_quadruples_new200.csv`  *(intermediate)*
Mistral-Nemo extraction on the 200 top-up reviews (**664 quads / 177 reviews**;
23 failed). Same columns. *Purpose:* clean replacements for the 140 failures.

### `Mistral_predicted_quadruples_final.csv`  ⭐ **production**
The assembled clean dataset: **860 good originals + 140 good from the top-up =
exactly 1,000 reviews, all with ≥1 quad, 3,951 quads.** Empty placeholders dropped;
2 off-taxonomy categories remapped to "Other" (no `parse_ok` column).
**Columns:** `review_id, facility_type_en, aspect_category, aspect_term,
opinion_term, sentiment, english_translated`. *Purpose:* the Mistral system output
scored in Stage 3.

> The three `Mistral_predicted_quadruples*` files are stages of ONE extraction:
> raw first pass → top-up → assembled clean 1,000. **Use `_final` as the dataset;**
> the other two are provenance/reproducibility.

---

## Stage 3 — Evaluation set & annotations

**Design:** the **human annotation is the gold standard**; three off-the-shelf LLM
annotations — **Mistral-Nemo, ChatGPT (GPT-5.5), Claude** — are each compared
against it via F1 + SAFE.

### `sample_100.csv`
The **100-review evaluation sample**, stratified to cover **all 43 facility types**.
**Columns:** `review_id, facility_type_en, content` (English). *Purpose:* the fixed
set every annotator/model labels.

### `annotation_template_100.csv`
Blank fill-in template = `sample_100` + empty quad columns. **Columns:**
`review_id, facility_type_en, content, aspect_category, aspect_term, opinion_term,
sentiment, notes`. *Purpose:* what a human (or LLM) fills in; one row per quad.

### `chatgpt_100_raw.csv`  *(intermediate)*
Raw export of the **ChatGPT (GPT-5.5, Medium)** annotation sheet — messy
(cp1252, duplicate blocks, ragged columns). *Purpose:* input to the cleaner.

### `chatgpt_100_annotation.csv`  ⭐ **GPT annotation (1 of 3 LLMs)**
Cleaned GPT-5.5 annotation: **601 quads over 96 reviews** (`stages/clean_annotation.py`
dedups, fixes encoding/malformed rows). **Columns:** `review_id, aspect_category,
aspect_term, opinion_term, sentiment, notes`.
*Note:* GPT-5.5 left 3 reviews unannotated (r020603, r022386, r064341); r066172 has
no aspects → covers **96/100**. This is a **system under test, not the gold.**

### `mistral_100_annotation.csv`  ⭐ **Mistral annotation (1 of 3 LLMs)**
Mistral-Nemo's quads for the same 100 sample reviews, built by
`stages/make_mistral_100.py` from the **original** run (`Mistral_predicted_quadruples.csv`)
— **not** `_final`, which had swapped out 15 of the sample ids. **311 quads over 85
reviews;** 15 sample reviews parse-failed → no rows. Columns match
`chatgpt_100_annotation.csv`.

> **Coverage of the 100-review eval set (align before Stage 3):** Mistral **85/100**
> · ChatGPT **96/100** · Claude *(pending)* · Human *(pending)*. `r066172` genuinely
> has no aspects. A review a system misses counts as a recall miss in scoring — keep
> the eval set fixed at the **100 sample ids**, don't shrink it to a system's coverage.

### *(pending)*
- `claude_100_annotation.csv` — Claude annotation (needs Anthropic API billing)
- `human_100_annotation.csv` — the **human gold standard**
- Both use the same template and are cleaned by `stages/clean_annotation.py --src … --out …`.

---

## Notes
- `data/interim/` (checkpoints) and `archive/`, `scratch/`, `*.log` are gitignored.
- Aspect taxonomy: `config.ASPECT_CATEGORIES` (6 + Other). Annotation rules:
  `docs/annotation_guidelines.md`.
