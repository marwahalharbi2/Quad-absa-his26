"""
Central configuration for the Paper 6 Healthcare Quad-ABSA pipeline.

All stage parameters live here so runs are reproducible and easy to audit.
Values follow Paper_6_Pipeline_Instructions_for_Claude_Code.md.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
OUTPUT = DATA / "output"
REPORTS = ROOT / "reports"

for _p in (RAW, INTERIM, OUTPUT, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Stage -1: Translation (Russian -> English)
# ----------------------------------------------------------------------------
STAGE_M1 = {
    "dataset_name": "blinoff/healthcare_facilities_reviews",
    "source_lang": "ru",
    "target_lang": "en",
    "model_name": "facebook/m2m100_418M",   # _1.2B for slightly better quality
    "batch_size": 16,                        # lowered for CPU friendliness
    "max_input_tokens": 512,                 # truncate very long reviews
    "sample_size_for_validation": 50,
    # QA scores are BERTScore round-trip (ru->en->ru) rescaled against an
    # empirical baseline of random ru pairs, so random~0 and perfect~1.
    # Threshold below is on the *rescaled* score. Set to 0.30 from the 50-sample
    # distribution: genuinely broken translations fall below ~0.30; this is a
    # flag-and-proceed signal, not a hard gate (per plan). Mean rescaled ~0.51.
    "quality_threshold": 0.30,
    "baseline_pairs": 30,                    # random ru pairs for baseline estimate
    "device": "auto",                        # auto -> cuda if available else cpu
    "bertscore_model": "bert-base-multilingual-cased",  # for ru back-translation check
    "seed": 42,
}

# ----------------------------------------------------------------------------
# Facility-type translation map (ru -> en). The source dataset's `category`
# column is the FACILITY TYPE (dental clinic, maternity hospital, ...), not an
# ABSA aspect category — it is renamed `facility_type`. The word "category" is
# reserved for the aspect category the LLM extracts in Quad-ABSA.
# ----------------------------------------------------------------------------
FACILITY_TYPE_EN = {
    "Амбулатории, медпункты, здравпункты": "Outpatient clinics & health posts",
    "Аптеки": "Pharmacies",
    "Больницы": "Hospitals",
    "Больницы ведомственные": "Departmental hospitals",
    "Больницы гинекологические": "Gynecological hospitals",
    "Больницы городские": "City hospitals",
    "Больницы детские": "Children's hospitals",
    "Больницы инфекционные": "Infectious disease hospitals",
    "Больницы наркологические": "Addiction treatment hospitals",
    "Больницы онкологические": "Oncology hospitals",
    "Больницы ортопедо-хирургические": "Orthopedic surgery hospitals",
    "Больницы офтальмологические": "Ophthalmology hospitals",
    "Больницы психиатрические": "Psychiatric hospitals",
    "Больницы психоневрологические": "Psychoneurological hospitals",
    "Больницы туберкулезные": "Tuberculosis hospitals",
    "Водительские комиссии": "Driver medical examination boards",
    "Военные госпитали": "Military hospitals",
    "Гинекологические клиники центры репродукции": "Gynecology clinics & reproduction centers",
    "Детские медицинские центры": "Children's medical centers",
    "Диспансеры кожно-венерологические": "Dermatovenerology dispensaries",
    "Диспансеры наркологические": "Addiction treatment dispensaries",
    "Диспансеры онкологические": "Oncology dispensaries",
    "Диспансеры психоневрологические": "Psychoneurological dispensaries",
    "Женские консультации": "Women's health (antenatal) clinics",
    "Клиники и лечебные центры при медицинских НИИ": "Clinics at medical research institutes",
    "Лечебно-оздоровительные центры": "Medical & wellness centers",
    "Логопедические центры": "Speech therapy centers",
    "Магазины ортопедических изделий протезов": "Orthopedic & prosthetics stores",
    "Магазины товаров для здоровья": "Health products stores",
    "Магазины товаров для инвалидов": "Disability aids stores",
    "Медико-социальная экспертиза": "Medical-social (disability) assessment",
    "Медицинские анализы лаборатории": "Medical testing laboratories",
    "Медицинские центры": "Medical centers",
    "Медсанчасти": "Medical units (medsanchast)",
    "Наркологические клиники лечение алкоголизма": "Addiction clinics (alcoholism treatment)",
    "Ортопедические центры лечение позвоночника и суставов": "Orthopedic centers (spine & joints)",
    "Офтальмологические клиники": "Ophthalmology clinics",
    "Переливание крови сдача крови": "Blood transfusion & donation",
    "Поликлиники ведомственные": "Departmental polyclinics",
    "Поликлиники детские": "Children's polyclinics",
    "Поликлиники стоматологические": "Dental polyclinics",
    "Родильные дома": "Maternity hospitals",
    "Салоны и магазины очков": "Eyewear stores & salons",
    "Скорая медицинская помощь": "Emergency medical services",
    "Стоматологические клиники": "Dental clinics",
    "Травмпункты": "Emergency trauma centers",
    "Центры магнитно-резонансной томографии": "MRI centers",
    "Центры пластической хирургии и косметологии": "Plastic surgery & cosmetology centers",
}

# ----------------------------------------------------------------------------
# Quad-ABSA aspect categories (the LLM's target taxonomy).
# Derived from LDA discovery + human consolidation (see analysis/lda_topics.py).
# "Other" is the fallback when an aspect fits none of the six.
# ----------------------------------------------------------------------------
ASPECT_CATEGORIES = {
    "Staff & Service":
        "Interpersonal care, professionalism, attitude, communication, "
        "friendliness, and overall interaction quality with staff/doctors.",
    "Specialised Services":
        "Domain-specific medical services: dental, obstetrics/maternity, "
        "gynecology, surgery/operations, ultrasound/diagnostics, lab tests.",
    "Clinical & Treatment":
        "Diagnosis, prescriptions, treatment protocols, procedures, and "
        "clinical outcomes (whether the care worked).",
    "Operational Experience":
        "Waiting times, queues, admission, registration, referrals, "
        "appointment booking, and general accessibility/logistics.",
    "Infrastructure & Resources":
        "Equipment, facilities, cleanliness, physical environment/condition, "
        "and resource/pharmacy availability.",
    "Financial & Administrative":
        "Cost, price, insurance, payment, billing, paperwork, and policy "
        "communication.",
    "Other":
        "Any aspect that does not fit the six categories above.",
}

# ----------------------------------------------------------------------------
# Stage 0: LDA topic discovery
# ----------------------------------------------------------------------------
STAGE_0 = {
    "k_topics": 18,
    "beta": 0.1,
    "num_iterations": 1500,
    "random_state": 42,
    "top_n_words": 15,
    "min_word_freq": 2,
    "min_doc_length": 10,
    "healthcare_stopwords": [
        "facility", "center", "care", "place", "room", "area",
        "clinic", "hospital", "doctor", "patient", "review",
    ],
}

# ----------------------------------------------------------------------------
# Stage 1: Annotation
# ----------------------------------------------------------------------------
STAGE_1 = {
    "num_reviews_to_annotate": 250,
    "sample_strategy": "stratified",   # or "random"
    "num_annotators": 1,
    "inter_rater_threshold": 0.70,
    "random_state": 42,
}

# ----------------------------------------------------------------------------
# Stage 2: LLM generation
# ----------------------------------------------------------------------------
STAGE_2 = {
    "llm_model": "claude-3-5-sonnet-20241022",
    "temperature": 0.7,
    "max_tokens": 1000,
    "batch_size": 20,
    "api_timeout": 30,
    "retry_limit": 3,
}

# ----------------------------------------------------------------------------
# Stage 3: Evaluation
# ----------------------------------------------------------------------------
STAGE_3 = {
    "safe_threshold": 0.70,
    "bertscore_model": "bert-base-uncased",
    "bertscore_lang": "en",
    "aspect_weight": 0.25,
    "aspect_term_weight": 0.25,
    "opinion_term_weight": 0.25,
    "sentiment_weight": 0.25,
}
