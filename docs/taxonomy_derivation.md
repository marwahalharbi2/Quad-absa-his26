# Taxonomy Derivation (Stage 0: LDA → Six Categories + Other)

This document records how the aspect-category taxonomy used throughout the
pipeline was derived: from LDA topics, through hand-labelled aspect themes, to the
locked **six categories + Other**. It is the trail behind Section 3.2 of the paper.

```
30 LDA topics  ──(label each topic)──►  11 aspect themes  ──(consolidate)──►  6 categories + Other
lda_topics_30.csv                        lda_aspect_themes.csv                  config.ASPECT_CATEGORIES
```

---

## 1. Step 1 — LDA topic discovery

| Setting | Value |
|---|---|
| Script | `analysis/lda_topics.py` |
| Command | `python analysis/lda_topics.py --k 30` |
| Input | `data/output/absa_input.csv`, column `english_translated` (the 1,000-review translated sample) |
| Tokenisation | gensim `simple_preprocess` (accents stripped, tokens ≥ 3 chars) |
| Stop words | scikit-learn English list + healthcare stop words (`config.STAGE_0["healthcare_stopwords"]` plus generic review verbs such as *doctor, clinic, hospital, told, came*) |
| Document filter | documents with < 10 tokens after preprocessing are dropped |
| Vocabulary filter | `filter_extremes(no_below=2, no_above=0.5)` |
| Model | gensim `LdaModel`, K = 30, `passes=15`, `iterations=200`, `alpha="auto"`, `eta="auto"`, `random_state=42` |
| Output | `data/output/lda_topics_30.csv` — `topic_id`, per-topic `c_v` coherence, top 15 words, hand-assigned `theme` label |

Per-topic `c_v` coherence ranges from 0.16 to 0.44 (mean 0.26). Topics were read
as candidate aspects, not as final categories; several are mixed, which is why
the next two steps are interpretive.

The `config.STAGE_0["k_topics"]` value is not used: K is set on the command line.

## 2. Step 2 — Topics → aspect themes

Each topic was read from its top words and given a short label (`theme` column of
`lda_topics_30.csv`). Topics describing the same aspect were then grouped into
11 aspect themes (`data/output/lda_aspect_themes.csv`).

A topic may support more than one theme when its top words mix two aspects
(e.g. T4 mixes dental words with *money, pay, contract*). This overlap is
deliberate: 8 topics (T3, T4, T7, T16, T22, T25, T26, T30) appear under two
themes.

| Aspect theme | LDA topics |
|---|---|
| Staff attitude / care / professionalism | T7, T16, T18, T26 |
| Dental | T3, T4, T24 |
| Maternity & childbirth | T21, T22, T29 |
| Gynecology / antenatal / ultrasound | T5, T10, T14, T17, T30 |
| Cost / payment / insurance | T2, T4, T6, T23, T25 |
| Waiting times / queues | T3, T9, T16, T30 |
| Registration / admin / referrals | T8, T12, T27 |
| Surgery / operations | T7, T11, T22, T26 |
| Diagnosis / treatment / prescriptions | T13, T28 |
| Equipment / service quality | T20 |
| Pharmacy / dermatology / other | T15, T25 |

T1 (ophthalmology & paediatric diagnosis, mixed) and T19 (general gratitude for
long-term care, no distinct aspect) were not assigned to a theme.

## 3. Step 3 — Aspect themes → six categories + Other

The 11 themes were consolidated by domain-expert refinement into six categories.
The guiding rule was to keep labels **generalised rather than healthcare-specific**,
so that most categories carry over to other service domains.

| Category | Aspect themes consolidated into it | LDA topics |
|---|---|---|
| **Staff & Service** | Staff attitude / care / professionalism | T7, T16, T18, T26 |
| **Specialised Services** | Dental; Maternity & childbirth; Gynecology / antenatal / ultrasound; Surgery / operations | T3, T4, T5, T7, T10, T11, T14, T17, T21, T22, T24, T26, T29, T30 |
| **Clinical & Treatment** | Diagnosis / treatment / prescriptions; dermatology & medication part of *Pharmacy / dermatology / other* | T13, T15, T28 |
| **Operational Experience** | Waiting times / queues; Registration / admin / referrals | T3, T8, T9, T12, T16, T27, T30 |
| **Infrastructure & Resources** | Equipment / service quality; pharmacy part of *Pharmacy / dermatology / other* | T20, T25 |
| **Financial & Administrative** | Cost / payment / insurance | T2, T4, T6, T23, T25 |
| **Other** | Fallback for evaluated aspects that fit none of the six | — |

Full category definitions, with example cues and tie-break rules, are in
[`annotation_guidelines.md`](annotation_guidelines.md) §3 and in
`config.ASPECT_CATEGORIES` (the version given to every LLM annotator).

Category scope was widened beyond the topic evidence where the definition needed
it: for example, *Infrastructure & Resources* also covers cleanliness and the
physical environment, and *Financial & Administrative* also covers paperwork and
policy communication.

## 4. Locked taxonomy

After Step 3 the taxonomy was **locked**: the same seven labels and definitions
were used for the human gold standard (Stage 1), all three LLM annotators
(Stage 1), the model comparison (Stage 2), and full-corpus extraction (Stage 3).

## 5. Reusing the taxonomy in another domain

- **Domain-general (carry over unchanged):** Staff & Service, Operational
  Experience, Infrastructure & Resources, Financial & Administrative.
- **Domain-specific (re-instantiate per domain):** Specialised Services and
  Clinical & Treatment capture the domain's core offering. In a new domain, run
  Step 1 on that corpus and rebuild these two from its topics (e.g. for hotels,
  room/amenity services and stay outcome).
- **Other** stays as the fallback.
