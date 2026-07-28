"""
LDA topic model over the English reviews in absa_input.csv.

Kept as a BASELINE to compare LDA's unsupervised aspects against the LLM's
Quad-ABSA aspect categories later. Not a pipeline stage.

    python analysis/lda_topics.py --k 30

Output: data/output/lda_topics_<k>.csv  (topic_id, coherence, top_words)
Themes are added by hand afterwards (interpretation step).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import STAGE_0, OUTPUT  # noqa: E402

SRC = OUTPUT / "absa_input.csv"
TOP_N = STAGE_0["top_n_words"]          # 15
MIN_LEN = STAGE_0["min_doc_length"]     # 10 tokens
SEED = STAGE_0["random_state"]          # 42

DOMAIN_STOP = set(STAGE_0["healthcare_stopwords"]) | {
    "doctor", "clinic", "hospital", "patient", "medical", "nurse", "time",
    "day", "told", "came", "went", "did", "got", "just", "like", "said",
    "really", "make", "made", "want", "need", "know", "think", "thing",
}
STOP = ENGLISH_STOP_WORDS | DOMAIN_STOP


def tokenize(text: str) -> list[str]:
    return [t for t in simple_preprocess(str(text), deacc=True, min_len=3)
            if t not in STOP]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=30, help="number of topics")
    args = ap.parse_args()
    K = args.k

    df = pd.read_csv(SRC)
    docs = [tokenize(t) for t in df["english_translated"]]
    docs = [d for d in docs if len(d) >= MIN_LEN]
    print(f"[lda] {len(docs)} documents after preprocessing")

    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=STAGE_0["min_word_freq"], no_above=0.5)
    corpus = [dictionary.doc2bow(d) for d in docs]

    print(f"[lda] fitting LDA: K={K}, vocab={len(dictionary)} ...")
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=K,
                   random_state=SEED, passes=15, iterations=200,
                   alpha="auto", eta="auto")

    cm = CoherenceModel(model=lda, texts=docs, dictionary=dictionary, coherence="c_v")
    per_topic = cm.get_coherence_per_topic()
    print(f"[lda] overall coherence (c_v): {cm.get_coherence():.4f}")

    rows = []
    for tid in range(K):
        words = [w for w, _ in lda.show_topic(tid, topn=TOP_N)]
        rows.append({"topic_id": f"T{tid + 1}",
                     "coherence": round(per_topic[tid], 4),
                     "top_words": ", ".join(words)})
    out = pd.DataFrame(rows)
    out_path = OUTPUT / f"lda_topics_{K}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[lda] wrote {out_path}\n")
    for r in out.itertuples():
        print(f"  {r.topic_id} (coh {r.coherence:.2f}): {r.top_words}")


if __name__ == "__main__":
    main()
