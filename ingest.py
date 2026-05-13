#!/usr/bin/env python3
"""
Skill Selector — ingestion pipeline.

Usage:
  python ingest.py                    # seed run (anthropics + obra)
  python ingest.py --repos all        # all Strategy A hosted repos
  python ingest.py --repos all --linklist  # + Strategy B link-lists
  python ingest.py --skip-umap        # re-embed only, reuse existing UMAP
  python ingest.py --dry-run          # crawl + parse, no DB writes
"""
import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import psycopg2.extras
import umap as umap_lib

from db import get_conn
from embed import build_embed_text, embed_texts
from github import REPOS_HOSTED, REPOS_LINKLIST, SEED_REPOS, crawl_hosted, crawl_linklist

# ── paths ──────────────────────────────────────────────────────────────────────

BASE        = Path(__file__).parent
REDUCER_PKL = BASE / "umap_transform.pkl"
CACHE_FILE  = BASE / "data" / "skills_cache.json"

# ── OpenRouter client ──────────────────────────────────────────────────────────

def _openrouter_client():
    from openai import OpenAI
    from db import _load_secrets  # already called in db import but safe to re-call
    # Prefer direct OpenAI API (OpenRouter requires purchased credits)
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("ERROR: OPENAI_API_KEY or OPENROUTER_API_KEY not set")
    return OpenAI(api_key=key)


# ── crawl ──────────────────────────────────────────────────────────────────────

def collect_skills(repos: str, use_linklist: bool) -> list[dict]:
    """Crawl repos and return raw skill dicts. Dedup on (name, source_repo)."""
    seen: set[tuple] = set()
    skills: list[dict] = []

    # Determine which hosted repos to crawl
    if repos == "seed":
        hosted = [r for r in REPOS_HOSTED if f"{r['owner']}/{r['repo']}" in SEED_REPOS]
    elif repos == "all":
        hosted = REPOS_HOSTED
    else:
        slugs = repos.split(",")
        hosted = [r for r in REPOS_HOSTED if f"{r['owner']}/{r['repo']}" in slugs]

    for cfg in hosted:
        print(f"Crawling {cfg['owner']}/{cfg['repo']}...")
        for s in crawl_hosted(
            cfg["owner"], cfg["repo"], cfg.get("branch", "main"),
            cfg.get("skip_prefixes"), cfg.get("category_map"),
        ):
            key = (s["name"], s["source_repo"])
            if key not in seen:
                seen.add(key)
                skills.append(s)

    # Strategy B link-lists
    if use_linklist:
        already_seen_repos = {f"{r['owner']}/{r['repo']}" for r in hosted}
        for cfg in REPOS_LINKLIST:
            print(f"Link-list crawl: {cfg['owner']}/{cfg['repo']}...")
            for s in crawl_linklist(
                cfg["owner"], cfg["repo"], cfg.get("branch", "main"), already_seen_repos
            ):
                key = (s["name"], s["source_repo"])
                if key not in seen:
                    seen.add(key)
                    skills.append(s)

    # anthropics/skills wins name collisions
    anthropic_names = {
        s["name"] for s in skills if s["source_repo"] == "anthropics/skills"
    }
    deduped = []
    for s in skills:
        if s["source_repo"] != "anthropics/skills" and s["name"] in anthropic_names:
            continue
        deduped.append(s)

    print(f"Collected {len(deduped)} unique skills")
    return deduped


# ── embed ──────────────────────────────────────────────────────────────────────

def build_embeddings(skills: list[dict], client) -> np.ndarray:
    """Build embed_text for each skill and embed. Saves/loads cache."""
    CACHE_FILE.parent.mkdir(exist_ok=True)

    # Load existing cache
    cache: dict[str, list] = {}
    if CACHE_FILE.exists():
        cached = json.loads(CACHE_FILE.read_text())
        cache = {c["key"]: c for c in cached}

    # Build embed texts
    for s in skills:
        s["embed_text"], s["embed_tier"] = build_embed_text(
            s["description"], s.get("body", ""), s.get("tags")
        )

    # Identify uncached
    to_embed = [s for s in skills if s["embed_text"] not in cache]
    if to_embed:
        print(f"Embedding {len(to_embed)} new skills...")
        texts = [s["embed_text"] for s in to_embed]
        vecs = embed_texts(texts, client)
        for s, vec in zip(to_embed, vecs):
            cache[s["embed_text"]] = {"key": s["embed_text"], "vec": vec.tolist()}
        CACHE_FILE.write_text(json.dumps(list(cache.values())))
    else:
        print("All embeddings cached.")

    vectors = np.array([cache[s["embed_text"]]["vec"] for s in skills], dtype=np.float32)
    return vectors


# ── UMAP ───────────────────────────────────────────────────────────────────────

def fit_umap(vectors: np.ndarray, skip_umap: bool) -> tuple[np.ndarray, object]:
    """Fit or load UMAP reducer. Returns (coords_3d, reducer)."""
    if skip_umap and REDUCER_PKL.exists():
        print("Loading existing UMAP reducer...")
        with open(REDUCER_PKL, "rb") as f:
            reducer = pickle.load(f)
        coords = reducer.transform(vectors)
    else:
        print(f"Fitting UMAP on {len(vectors)} points...")
        reducer = umap_lib.UMAP(
            n_components=3, n_neighbors=15, min_dist=0.1, random_state=42
        )
        coords = reducer.fit_transform(vectors)
        with open(REDUCER_PKL, "wb") as f:
            pickle.dump(reducer, f)
        print(f"UMAP reducer saved to {REDUCER_PKL}")
    return coords.astype(np.float32), reducer


def project_query(text: str, client) -> np.ndarray:
    """Project a query string to 3D using the saved reducer."""
    from embed import embed_texts as _embed
    if not REDUCER_PKL.exists():
        raise RuntimeError("umap_transform.pkl not found — run ingest first")
    with open(REDUCER_PKL, "rb") as f:
        reducer = pickle.load(f)
    vec = _embed([text], client)
    return reducer.transform(vec).astype(np.float32)[0]


# ── DB write ───────────────────────────────────────────────────────────────────

def write_to_db(skills: list[dict], vectors: np.ndarray, coords: np.ndarray, dry_run: bool):
    if dry_run:
        print("[dry-run] would write", len(skills), "skills")
        for s in skills[:3]:
            print(f"  {s['name']} ({s['source_repo']}) tier={s['embed_tier']} size={s['size']}")
        return

    conn = get_conn()
    cur = conn.cursor()

    print(f"Writing {len(skills)} skills to DB...")
    inserted = updated = 0

    def _clean_str(v):
        """Strip NUL bytes and surrogate chars that PostgreSQL rejects."""
        if v is None:
            return ""
        s = str(v)
        # Multiple passes — some encodings produce \x00 during re-encode
        s = s.replace("\x00", "").replace("\u0000", "")
        s = s.encode("utf-8", "replace").decode("utf-8")
        s = s.replace("\x00", "").replace("\u0000", "")
        return s

    for s, vec, coord in zip(skills, vectors, coords):
        x, y, z = float(coord[0]), float(coord[1]), float(coord[2])
        # Guard against NaN coords from UMAP
        if any(v != v for v in (x, y, z)):  # NaN check
            x, y, z = 0.0, 0.0, 0.0
        geom = f"SRID=0;POINTZ({x} {y} {z})"
        vec_list = vec.tolist()

        cur.execute("""
            INSERT INTO skill_selector.skills
                (name, url, source_repo, category, description, body, author,
                 size, char_count, embed_text, embed_tier, domain_inferred,
                 embedding, point_3d, updated_at)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s,
                 %s, %s, %s, %s, false,
                 %s, ST_GeomFromEWKT(%s), now())
            ON CONFLICT (name, source_repo) DO UPDATE SET
                url         = EXCLUDED.url,
                category    = EXCLUDED.category,
                description = EXCLUDED.description,
                body        = EXCLUDED.body,
                size        = EXCLUDED.size,
                char_count  = EXCLUDED.char_count,
                embed_text  = EXCLUDED.embed_text,
                embed_tier  = EXCLUDED.embed_tier,
                embedding   = EXCLUDED.embedding,
                point_3d    = EXCLUDED.point_3d,
                updated_at  = now()
            RETURNING (xmax = 0) AS inserted
        """, (
            _clean_str(s["name"]), _clean_str(s["url"]), _clean_str(s["source_repo"]),
            _clean_str(s["category"]), _clean_str(s["description"]),
            _clean_str(s.get("body", "")), _clean_str(s.get("author", "")),
            s["size"], s["char_count"],
            _clean_str(s["embed_text"]), s["embed_tier"],
            vec_list, geom,
        ))
        row = cur.fetchone()
        if row and row["inserted"]:
            inserted += 1
        else:
            updated += 1

    conn.commit()
    print(f"  inserted={inserted} updated={updated}")

    # Domains
    _update_domains(conn, cur)
    conn.commit()
    conn.close()


def _update_domains(conn, cur):
    """Recompute domain centroids, density, skill_count, local UMAP."""
    print("Updating domains...")

    cur.execute("""
        SELECT DISTINCT category FROM skill_selector.skills
        WHERE category IS NOT NULL AND category != 'Uncategorised'
    """)
    categories = [r["category"] for r in cur.fetchall()]

    for cat in categories:
        cur.execute("""
            SELECT id, embedding, point_3d
            FROM skill_selector.skills
            WHERE category = %s
        """, (cat,))
        rows = cur.fetchall()
        if not rows:
            continue

        skill_ids = [r["id"] for r in rows]
        # pgvector returns embeddings as strings; parse if needed
        def _parse_vec(v):
            if isinstance(v, str):
                return [float(x) for x in v.strip('[]').split(',')]
            return v
        vecs = np.array([_parse_vec(r["embedding"]) for r in rows], dtype=np.float32)
        centroid = vecs.mean(axis=0)

        # Density: mean pairwise cosine distance (sampled for large clusters)
        sample = vecs[:50]
        norms = np.linalg.norm(sample, axis=1, keepdims=True) + 1e-9
        normed = sample / norms
        sims = normed @ normed.T
        dists = 1.0 - sims
        np.fill_diagonal(dists, 0)
        density = float(dists.sum() / max(len(sample) * (len(sample) - 1), 1))

        # Centroid 3D — mean of ST_X/Y/Z
        cur.execute("""
            SELECT AVG(ST_X(point_3d)) AS cx,
                   AVG(ST_Y(point_3d)) AS cy,
                   AVG(ST_Z(point_3d)) AS cz
            FROM skill_selector.skills
            WHERE category = %s AND point_3d IS NOT NULL
        """, (cat,))
        c3 = cur.fetchone()
        cx, cy, cz = (c3["cx"] or 0), (c3["cy"] or 0), (c3["cz"] or 0)
        centroid_geom = f"SRID=0;POINTZ({cx} {cy} {cz})"

        cur.execute("""
            INSERT INTO skill_selector.domains (name, centroid, centroid_3d, density, skill_count)
            VALUES (%s, %s, ST_GeomFromEWKT(%s), %s, %s)
            ON CONFLICT (name) DO UPDATE SET
                centroid    = EXCLUDED.centroid,
                centroid_3d = EXCLUDED.centroid_3d,
                density     = EXCLUDED.density,
                skill_count = EXCLUDED.skill_count
        """, (cat, centroid.tolist(), centroid_geom, density, len(skill_ids)))

        # Local UMAP (only if enough points; spectral init needs N >> n_components)
        if len(vecs) >= 8:
            n_neighbors = min(15, len(vecs) - 1)
            local_reducer = umap_lib.UMAP(
                n_components=3, n_neighbors=n_neighbors, min_dist=0.1,
                random_state=42, init="random",
            )
            try:
                local_coords = local_reducer.fit_transform(vecs).astype(np.float32)
            except Exception as e:
                print(f"  local UMAP failed for {cat} ({len(vecs)} pts): {e}")
                local_coords = vecs[:, :3]
        else:
            local_coords = vecs[:, :3]  # trivial fallback for tiny domains

        cur.execute("""
            SELECT id FROM skill_selector.domains WHERE name = %s
        """, (cat,))
        domain_id = cur.fetchone()["id"]

        for skill_id, lc in zip(skill_ids, local_coords):
            lx, ly, lz = float(lc[0]), float(lc[1]), float(lc[2])
            lgeom = f"SRID=0;POINTZ({lx} {ly} {lz})"
            cur.execute("""
                INSERT INTO skill_selector.skill_domains (skill_id, domain_id, point_3d_local)
                VALUES (%s, %s, ST_GeomFromEWKT(%s))
                ON CONFLICT (skill_id, domain_id) DO UPDATE SET
                    point_3d_local = EXCLUDED.point_3d_local
            """, (skill_id, domain_id, lgeom))

    print(f"  {len(categories)} domains updated")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", default="seed",
                    help="seed | all | comma-separated owner/repo slugs")
    ap.add_argument("--linklist", action="store_true",
                    help="also crawl link-list repos (Strategy B)")
    ap.add_argument("--skip-umap", action="store_true",
                    help="skip UMAP refit, transform with existing reducer")
    ap.add_argument("--dry-run", action="store_true",
                    help="crawl + embed but do not write to DB")
    args = ap.parse_args()

    client = _openrouter_client()

    skills = collect_skills(args.repos, args.linklist)
    if not skills:
        print("No skills found.")
        return

    vectors = build_embeddings(skills, client)
    coords, _reducer = fit_umap(vectors, args.skip_umap)
    write_to_db(skills, vectors, coords, args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()
