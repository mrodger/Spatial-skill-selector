"""
tag_skills.py — assign 3 TF-IDF tags per skill deterministically.

Method:
  - Corpus = embed_text for all skills (the already-processed tiered text)
  - TfidfVectorizer with unigrams + bigrams, filtered by df bounds
  - Top-3 terms by TF-IDF weight per document → stored in skills.tags (text[])

Run: python tag_skills.py [--dry-run]
"""

import sys
import re
import psycopg2.extras
from sklearn.feature_extraction.text import TfidfVectorizer

from db import get_conn

N_TAGS = 3

# Terms that slip past stopwords but are useless as tags
EXTRA_STOP = {
    "use", "used", "uses", "using", "user", "users",
    "tool", "tools", "skill", "skills",
    "task", "tasks", "data", "output", "input",
    "work", "works", "make", "makes", "help", "helps",
    "provide", "provides", "support", "supports",
    "include", "includes", "available", "need", "needs",
    "allow", "allows", "like", "based", "given",
    "able", "want", "wants", "way", "ways",
}


def clean(text: str) -> str:
    """Light normalisation — lowercase, strip punctuation runs."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_tags(dry_run: bool = False):
    conn = get_conn()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Add column if missing
    cur.execute("""
        ALTER TABLE skill_selector.skills
        ADD COLUMN IF NOT EXISTS tags text[];
    """)
    conn.commit()

    # Load corpus
    cur.execute("SELECT id, embed_text FROM skill_selector.skills ORDER BY id")
    rows = cur.fetchall()
    ids   = [r["id"]         for r in rows]
    texts = [clean(r["embed_text"]) for r in rows]

    print(f"Corpus: {len(texts)} documents")

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,          # must appear in ≥2 docs (filters noise/typos)
        max_df=0.6,        # must appear in ≤60% of docs (filters corpus-wide filler)
        max_features=8000,
        stop_words="english",
        token_pattern=r"(?u)\b[a-z][a-z]{2,}\b",  # letters only, ≥3 chars
        sublinear_tf=True,
    )
    tfidf_matrix = vec.fit_transform(texts)
    feature_names = vec.get_feature_names_out()

    updates = []
    for i, doc_id in enumerate(ids):
        row_vec   = tfidf_matrix[i].toarray()[0]
        top_idx   = row_vec.argsort()[::-1]
        tags = []
        for idx in top_idx:
            if len(tags) >= N_TAGS:
                break
            term = feature_names[idx]
            # Skip if any token is in the extra stoplist
            if any(t in EXTRA_STOP for t in term.split()):
                continue
            tags.append(term)
        updates.append((tags, doc_id))

    if dry_run:
        for tags, doc_id in updates[:20]:
            print(f"  {doc_id}: {tags}")
        print(f"  ... {len(updates)} total (dry run, no write)")
        conn.close()
        return

    psycopg2.extras.execute_batch(cur,
        "UPDATE skill_selector.skills SET tags = %s WHERE id = %s",
        updates, page_size=200,
    )
    conn.commit()
    conn.close()
    print(f"Tagged {len(updates)} skills.")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    build_tags(dry_run)
