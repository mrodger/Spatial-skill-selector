#!/usr/bin/env python3
"""
Assign categories to Uncategorised skills by cosine similarity to named-domain centroids.

Usage:
  python reclassify.py              # reclassify all Uncategorised skills
  python reclassify.py --dry-run    # show what would change, no writes
  python reclassify.py --min-score 0.3  # only assign if cosine >= threshold
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from db import get_conn


def _parse_vec(v):
    if isinstance(v, str):
        return np.array([float(x) for x in v.strip("[]").split(",")], dtype=np.float32)
    return np.array(v, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def reclassify(dry_run: bool = False, min_score: float = 0.25):
    conn = get_conn()
    cur = conn.cursor()

    # Load named domain centroids (exclude Uncategorised)
    cur.execute("""
        SELECT name, centroid FROM skill_selector.domains
        WHERE name != 'Uncategorised' AND centroid IS NOT NULL
    """)
    domains = cur.fetchall()
    if not domains:
        print("No named domains found — run ingest first.")
        conn.close()
        return

    domain_names = [d["name"] for d in domains]
    domain_vecs = [_parse_vec(d["centroid"]) for d in domains]
    print(f"Loaded {len(domain_names)} named domains")

    # Load all Uncategorised skills with embeddings
    cur.execute("""
        SELECT id, name, embedding FROM skill_selector.skills
        WHERE category = 'Uncategorised' AND embedding IS NOT NULL
    """)
    skills = cur.fetchall()
    print(f"Reclassifying {len(skills)} Uncategorised skills...")

    assignments = {}  # category -> count
    low_confidence = 0

    for s in skills:
        vec = _parse_vec(s["embedding"])
        scores = [_cosine(vec, dv) for dv in domain_vecs]
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]

        if best_score < min_score:
            low_confidence += 1
            continue

        cat = domain_names[best_idx]
        assignments[cat] = assignments.get(cat, 0) + 1

        if not dry_run:
            cur.execute(
                "UPDATE skill_selector.skills SET category = %s, domain_inferred = true WHERE id = %s",
                (cat, s["id"]),
            )

    conn.commit()

    print(f"\nAssignments (min_score={min_score}):")
    for cat, n in sorted(assignments.items(), key=lambda x: -x[1]):
        print(f"  {n:4d}  {cat}")
    print(f"  {low_confidence:4d}  (below threshold — left as Uncategorised)")
    if dry_run:
        print("\n[dry-run] no changes written")

    # After reclassifying, rebuild domain centroids if not dry run
    if not dry_run:
        print("\nRebuilding domain table...")
        cur.execute("""
            SELECT DISTINCT category FROM skill_selector.skills
            WHERE category != 'Uncategorised'
        """)
        cats = [r["category"] for r in cur.fetchall()]
        for cat in cats:
            cur.execute("""
                SELECT embedding FROM skill_selector.skills WHERE category = %s
            """, (cat,))
            rows = cur.fetchall()
            vecs = np.array([_parse_vec(r["embedding"]) for r in rows], dtype=np.float32)
            centroid = vecs.mean(axis=0)
            cur.execute("""
                INSERT INTO skill_selector.domains (name, centroid, skill_count)
                VALUES (%s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                    centroid = EXCLUDED.centroid,
                    skill_count = EXCLUDED.skill_count
            """, (cat, centroid.tolist(), len(rows)))
        conn.commit()
        print(f"  {len(cats)} domains updated")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-score", type=float, default=0.25,
                    help="Minimum cosine similarity to assign a category (default 0.25)")
    args = ap.parse_args()
    reclassify(dry_run=args.dry_run, min_score=args.min_score)
