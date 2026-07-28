# Quad-ABSA Annotation Guidelines (Healthcare Reviews)

**Task:** manually annotate 100 English healthcare-facility reviews with
**Quad-ABSA quadruples**, to build a human *gold standard*. This gold standard
is later compared against the LLM's automatic extraction (F1 + SAFE metrics).

**Files**
- Reviews to annotate: `data/output/sample_100.csv` (`review_id`, `facility_type_en`, `content`)
- Fill your answers into: `data/output/annotation_template_100.csv`

**Effort:** ~3–5 minutes per review (~5–8 hours total for 100).

---

## 1. What you are extracting — the quadruple

For each **aspect** a reviewer expresses an opinion about, record a 4-part tuple:

| Element | Meaning |
|---|---|
| **aspect_category** | which of the 7 categories the aspect belongs to (Section 3) |
| **aspect_term** | the phrase in the review that *gives away* the aspect (Section 4) |
| **opinion_term** | the phrase in the review that expresses the *evaluation* (Section 5) |
| **sentiment** | positive / negative / neutral (Section 6) |

A review can contain **0, 1, or many** aspects → **one row per quadruple**.

**Anchor example:**
Review: *"It takes ages to be admitted to the ER, but the doctors were kind."*
→ (Operational Experience | **be admitted** | **takes ages** | negative)
→ (Staff & Service | **doctors** | **kind** | positive)

**The counting unit is the (aspect, opinion) PAIR — never merge shared aspects or opinions:**
- **Two aspects sharing one opinion → two quadruples** (repeat the opinion span in each).
  *"the doctors **and** nurses were **rude**"* → (Staff & Service | doctors | rude | negative) **and** (Staff & Service | nurses | rude | negative).
- **One aspect with two opinions → two quadruples** (repeat the aspect span in each).
  *"the **service** is **cheap** but **slow**"* → (… | service | cheap | positive) **and** (… | service | slow | negative).

This is how a person naturally reads the sentence — do **not** collapse a shared aspect or a shared opinion into a single row.

---

## 2. The annotation process (step by step)

1. **Read the whole review** first — don't annotate sentence by sentence blindly.
2. **Find every aspect the reviewer evaluates** (praises, complains about, or
   states a fact with a clear positive/negative slant).
3. For **each** aspect, in one row of the template, fill:
   - `aspect_category` — pick exactly one (Section 3)
   - `aspect_term` — copy the giving-away phrase from the text (Section 4)
   - `opinion_term` — copy the evaluating phrase from the text (Section 5)
   - `sentiment` — positive / negative / neutral (Section 6)
4. If the review has **no evaluative content at all** (pure factual note, e.g.
   "The clinic is on Lenin Street"), record **no quadruples** — leave the review's
   quad fields blank and move on.
5. Add **extra rows** (copy the `review_id`) when a review has more than one aspect.

---

## 3. Aspect categories (choose exactly ONE per quadruple)

Use only these seven. If an aspect fits more than one, pick the **most specific /
dominant** one. Use **Other** only when none of the six genuinely fit.

| Category | Covers | Example cues |
|---|---|---|
| **Staff & Service** | interpersonal care, professionalism, attitude, communication, friendliness, competence of staff/doctors/nurses | "nurses", "Dr. Ivanov", "rude receptionist", "attentive", "explained everything" |
| **Specialised Services** | domain-specific medical service: dental, obstetrics/maternity, gynecology, surgery/operation, ultrasound/diagnostics, lab tests | "wisdom tooth removal", "the birth", "ultrasound", "the operation" |
| **Clinical & Treatment** | diagnosis, prescriptions, treatment protocol, procedures, and whether the care *worked* (outcome) | "diagnosis was wrong", "prescribed antibiotics", "the treatment helped", "recovered fully" |
| **Operational Experience** | waiting time, queues, admission, registration, referrals, appointment booking, accessibility, logistics | "waited two hours", "queue", "be admitted", "couldn't get an appointment", "sent us away" |
| **Infrastructure & Resources** | equipment, facilities, cleanliness, physical environment/condition, pharmacy stock | "old equipment", "clean rooms", "no beds", "modern building", "medicine out of stock" |
| **Financial & Administrative** | cost, price, insurance, payment, billing, paperwork, policy communication | "too expensive", "3000 rubles", "free of charge", "insurance not accepted", "paperwork" |
| **Other** | a genuine evaluated aspect that fits none of the six above | (use sparingly) |

> If you are torn between **Specialised Services** (the *what* — e.g. "the operation")
> and **Clinical & Treatment** (the *outcome* — e.g. "it cured me"), annotate the
> phrase as it appears: "the operation went well" → Specialised Services;
> "I recovered fully" → Clinical & Treatment.

---

## 4. aspect_term — the phrase that "gives away" the aspect

- **Copy the shortest phrase from the review** that identifies the aspect.
- It can be a **noun** ("nurses", "equipment", "price") **or an action/verb phrase**
  that reveals the aspect ("**be admitted**", "waiting for results", "getting a referral").
- Prefer the concrete phrase the reviewer actually used over a paraphrase.
- Use **`IMPLICIT`** only when the aspect is clearly evaluated but **no phrase in
  the text names or reveals it** (rare — try hard to find the giving-away phrase first).

**Examples**
- "The **nurses** were kind" → aspect_term = `nurses`
- "It takes ages **to be admitted**" → aspect_term = `be admitted`
- "Everything here is overpriced" → the aspect (cost) is revealed by "overpriced";
  aspect_term = `overpriced`? No — "overpriced" is the *opinion*. The aspect (price)
  isn't named separately → aspect_term = `IMPLICIT`, opinion_term = `overpriced`.

---

## 5. opinion_term — the phrase that expresses the evaluation

- **Copy the shortest phrase** that carries the sentiment: adjective, verb, or short
  phrase ("professional", "takes ages", "did everything great", "a disgrace").
- If the aspect is mentioned and clearly evaluated but there is **no explicit opinion
  word** (the sentiment is implied by context), use **`IMPLICIT`**.
- One aspect can have **multiple opinions** → make one quadruple per opinion.

**Examples**
- "the recovery room was **spacious**" → opinion_term = `spacious`
- "waited **two hours**" → opinion_term = `two hours` (or `waited two hours`) — the wait itself is the negative
- "They **refused to see** my child" → opinion_term = `refused to see`

---

## 6. sentiment — positive / negative / neutral

Assign the sentiment **the reviewer expresses toward that specific aspect**:

- **positive** — praise, satisfaction, gratitude ("wonderful", "thank you", "helped a lot").
- **negative** — complaint, dissatisfaction, harm ("rude", "too slow", "made it worse").
- **neutral** — the aspect is mentioned factually with no clear positive/negative slant,
  or a genuinely mixed/balanced statement. Use neutral **sparingly**.

Rules:
- Judge sentiment **per aspect**, not for the whole review. A review can be positive
  about staff and negative about waiting time.
- **Handle negation & sarcasm by intended meaning:** "not bad at all" → positive;
  "*great*, another two-hour wait" (sarcastic) → negative.
- A polite "thank you" attached to a specific person/service → positive for that aspect.

---

## 7. Decision rules & conventions

1. **Copy spans verbatim** from the English `content` (same words; you may trim
   leading/trailing articles). Do not translate, paraphrase, or fix grammar.
2. **One quadruple per (aspect, opinion) pair — the counting unit.** Two different
   aspects → two quadruples **even when they share one opinion word**: "the doctors
   *and* nurses were **rude**" → (…|doctors|rude|neg) **and** (…|nurses|rude|neg)
   (repeat the shared opinion span in each row).
3. **Same aspect, two opinions → two quadruples**, repeating the aspect span in each
   (e.g. "**service** is **cheap** but **slow**" → (…|service|cheap|pos) **and**
   (…|service|slow|neg)).
4. **Don't annotate non-evaluative facts** (dates, addresses, which floor). Only
   annotate aspects the reviewer *evaluates*.
5. **Whole-review sentiment with no named aspect** (e.g. "Best clinic ever!") →
   aspect_term = `IMPLICIT`, choose the best-fitting category (often Staff & Service
   or Other), opinion_term = the phrase ("Best clinic ever"), sentiment = positive.
6. **Named doctors/nurses** → Staff & Service, aspect_term = the name or role.
7. **Money amounts** ("3000 rubles", "free") → Financial & Administrative.
8. When **genuinely unsure** about a category, pick your best guess and add a note in
   the `notes` column — do not leave the category blank.
9. **Subject facility only.** Annotate only aspects of the facility *being reviewed*. If the
   reviewer mentions or compares against a **different** named facility (e.g. "…but
   hospital-32 was disgusting"), **do not** annotate opinions about that other facility —
   those belong to a different facility's reviews, not this one. (Same for opinions about a
   *previous* doctor/clinic used only as a contrast.)

---

## 8. Output format

Fill `data/output/annotation_template_100.csv`. **One row per quadruple.**

| Column | What to write |
|---|---|
| `review_id` | keep as given (copy the row to add more quads for the same review) |
| `facility_type_en` | keep as given (context only; do not change) |
| `content` | keep as given (the review text; do not change) |
| `aspect_category` | one of the 7 category names, spelled exactly |
| `aspect_term` | the giving-away phrase, or `IMPLICIT` |
| `opinion_term` | the evaluating phrase, or `IMPLICIT` |
| `sentiment` | `positive` / `negative` / `neutral` |
| `notes` | optional — flag uncertainty or tricky cases |

For a review with **no aspects**, leave `aspect_category`/`aspect_term`/
`opinion_term`/`sentiment` blank and write `no aspects` in `notes`.

---

## 9. Worked examples

**A.** *"Huge thanks to Dr. Matveev, the wisdom tooth removal took 10 seconds, painless!"*
- (Staff & Service | Dr. Matveev | Huge thanks | positive)
- (Specialised Services | wisdom tooth removal | took 10 seconds, painless | positive)

**B.** *"Terrible. Waited 40 minutes, the equipment is ancient, and they charged 3000 rubles for nothing."*
- (Operational Experience | Waited 40 minutes | Waited 40 minutes | negative)
- (Infrastructure & Resources | equipment | ancient | negative)
- (Financial & Administrative | charged 3000 rubles | for nothing | negative)

**C.** *"The staff at registration have no idea what respect means."*
- (Staff & Service | staff at registration | no idea what respect means | negative)

**D.** *"Came for an ultrasound, everything was explained clearly, recovered quickly."*
- (Specialised Services | ultrasound | IMPLICIT | neutral)
- (Staff & Service | explained | clearly | positive)
- (Clinical & Treatment | recovered | quickly | positive)

**E.** *"Best maternity hospital in the city!"*
- (Specialised Services | maternity hospital | Best ... in the city | positive)

---

## 10. Quality control

- **Consistency:** re-read Section 3 whenever a category feels ambiguous; keep your
  category choices consistent across the 100 reviews.
- **Two annotators (recommended):** annotate independently, then compute
  **inter-annotator agreement** (Cohen's Kappa; target ≥ 0.70) and resolve
  disagreements by discussion/consensus.
- **Pace:** aim for ~3–5 min/review; flag anything that took much longer in `notes`.
- **When in doubt, annotate what the text literally says** and note the doubt —
  do not invent aspects the reviewer did not evaluate.
