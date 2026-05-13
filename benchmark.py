#!/usr/bin/env python3
"""
Spatial vs semantic AB benchmark.
For each query, runs two independent searches against the DB:
  - semantic: global HNSW, pure cosine, top 3
  - spatial:  nearest cluster by 3D distance, ranked by pure 3D Euclidean, top 3
              (no cosine component — true spatial)

Outputs benchmark.json + a human-readable table.
"""
import json, pickle, os
import numpy as np
from pathlib import Path
from db import _load_secrets, get_conn
from embed import embed_texts

_load_secrets()

BASE        = Path(__file__).parent
REDUCER_PKL = BASE / "umap_transform.pkl"

QUERIES = [
    # direct — semantic should dominate
    ("direct",      "extract text and tables from a PDF"),
    ("direct",      "search the web for current information"),
    ("direct",      "send a Slack notification when a job finishes"),
    # indirect phrasing — vocabulary mismatch stress test
    ("indirect",    "I need something to keep tabs on what my agent gets up to while I'm not watching"),
    ("indirect",    "my agent keeps burning through my budget"),
    ("indirect",    "make sure Claude doesn't go rogue"),
    # goal-oriented — multi-step, spatial neighbourhood should help
    ("goal",        "generate 3D AI characters for a video"),
    ("goal",        "turn a research question into a finished written report"),
    ("goal",        "I want my users to be able to talk to my agent"),
    ("goal",        "help me understand what went wrong after my agent run"),
]

def _parse_vec(v):
    if isinstance(v, str):
        return np.array([float(x) for x in v.strip("[]").split(",")], dtype=np.float32)
    return np.array(v, dtype=np.float32)

def run_benchmark():
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
    client = OpenAI(api_key=key)

    with open(REDUCER_PKL, "rb") as f:
        reducer = pickle.load(f)

    conn = get_conn()
    cur  = conn.cursor()

    results = []

    for category, query in QUERIES:
        print(f"\n{'─'*60}")
        print(f"[{category.upper()}] {query}")

        # Embed
        vec = embed_texts([query], client)[0]
        q3d = reducer.transform([vec])[0]
        domain_3d = f"SRID=0;POINTZ({q3d[0]} {q3d[1]} {q3d[2]})"

        # ── Pure semantic: global HNSW, cosine only ───────────────────────
        cur.execute("""
            SELECT name, description,
                   1 - (embedding <=> %s::vector) AS cosine
            FROM skill_selector.skills
            ORDER BY embedding <=> %s::vector
            LIMIT 3
        """, (vec.tolist(), vec.tolist()))
        sem = [{"name": r["name"], "description": r["description"][:80],
                "score": round(float(r["cosine"]), 4)} for r in cur.fetchall()]

        # ── Pure spatial: nearest cluster by 3D dist, rank by 3D dist only ─
        # Step 1: find nearest cluster centroid by 3D Euclidean distance
        cur.execute("""
            SELECT name,
                   ST_Distance(centroid_3d, ST_GeomFromEWKT(%s)) AS dist_3d
            FROM skill_selector.domains
            WHERE centroid_3d IS NOT NULL
            ORDER BY centroid_3d <-> ST_GeomFromEWKT(%s)
            LIMIT 1
        """, (domain_3d, domain_3d))
        domain_row = cur.fetchone()
        domain_name = domain_row["name"] if domain_row else None
        domain_dist = round(float(domain_row["dist_3d"]), 4) if domain_row else None

        if domain_name:
            # Step 2: rank skills within that cluster by 3D distance to query — no cosine
            cur.execute("""
                SELECT s.name, s.description,
                       ST_Distance(s.point_3d, ST_GeomFromEWKT(%s)) AS dist_3d,
                       1 - (s.embedding <=> %s::vector) AS cosine
                FROM skill_selector.skills s
                JOIN skill_selector.skill_domains sd ON sd.skill_id = s.id
                JOIN skill_selector.domains d ON d.id = sd.domain_id
                WHERE d.name = %s
                ORDER BY s.point_3d <-> ST_GeomFromEWKT(%s)
                LIMIT 3
            """, (domain_3d, vec.tolist(), domain_name, domain_3d))
        else:
            # No cluster found — fall back to global 3D distance
            cur.execute("""
                SELECT name, description,
                       ST_Distance(point_3d, ST_GeomFromEWKT(%s)) AS dist_3d,
                       1 - (embedding <=> %s::vector) AS cosine
                FROM skill_selector.skills
                WHERE point_3d IS NOT NULL
                ORDER BY point_3d <-> ST_GeomFromEWKT(%s)
                LIMIT 3
            """, (domain_3d, vec.tolist(), domain_3d))

        spa = [{"name": r["name"], "description": r["description"][:80],
                "dist_3d": round(float(r["dist_3d"]), 4),
                "cosine":  round(float(r["cosine"]), 4)} for r in cur.fetchall()]

        # Overlap
        sem_names = {r["name"] for r in sem}
        spa_names = {r["name"] for r in spa}
        overlap   = sorted(sem_names & spa_names)
        diverge   = len(overlap) < 3

        # Print summary
        print(f"  Domain routed (3D): {domain_name}  (dist={domain_dist})")
        print(f"  Semantic top-3: {[r['name'] for r in sem]}")
        print(f"  Spatial  top-3: {[r['name'] for r in spa]}")
        print(f"  Overlap: {overlap or 'NONE'}  diverge={diverge}")
        if diverge:
            only_sem = sorted(sem_names - spa_names)
            only_spa = sorted(spa_names - sem_names)
            if only_sem: print(f"  Only semantic: {only_sem}")
            if only_spa: print(f"  Only spatial:  {only_spa}")

        results.append({
            "category":    category,
            "query":       query,
            "domain_3d":   domain_name,
            "domain_dist": domain_dist,
            "semantic":    sem,
            "spatial":     spa,
            "overlap":     overlap,
            "diverge":     diverge,
            "only_semantic": sorted(sem_names - spa_names),
            "only_spatial":  sorted(spa_names - sem_names),
        })

    conn.close()

    out = BASE / "benchmark.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n\nResults saved → {out}")

    # Summary table
    print(f"\n{'═'*60}")
    print(f"{'QUERY':<45} {'CAT':<8} {'OVERLAP':<5} {'DIV'}")
    print(f"{'─'*60}")
    for r in results:
        q = r["query"][:44]
        print(f"{q:<45} {r['category']:<8} {len(r['overlap'])}/3    {'YES' if r['diverge'] else 'no'}")

if __name__ == "__main__":
    run_benchmark()
