#!/usr/bin/env python3
"""Skill Selector — FastAPI server. Port 8200."""
import os
import pickle
from pathlib import Path

import numpy as np
import psycopg2.extras
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from db import _load_secrets, get_conn
from embed import EMBED_MODEL, embed_texts

_load_secrets()

BASE        = Path(__file__).parent
REDUCER_PKL = BASE / "umap_transform.pkl"
STATIC_DIR  = BASE / "static"

app = FastAPI(title="skill-selector")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Load UMAP reducer once at startup
_reducer = None

def _get_reducer():
    global _reducer
    if _reducer is None:
        if not REDUCER_PKL.exists():
            raise HTTPException(503, "UMAP reducer not found — run ingest.py first")
        with open(REDUCER_PKL, "rb") as f:
            _reducer = pickle.load(f)
    return _reducer


def _openrouter_client():
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
    return OpenAI(api_key=key)


# ── helpers ────────────────────────────────────────────────────────────────────

def _parse_vec(v) -> np.ndarray:
    """pgvector returns embeddings as '[f,f,...]' strings; parse if needed."""
    if isinstance(v, str):
        return np.array([float(x) for x in v.strip('[]').split(',')], dtype=np.float32)
    return np.array(v, dtype=np.float32)


def _cosine(a: np.ndarray, b) -> float:
    b = _parse_vec(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _chunk_text(text: str, size: int = 200) -> list[str]:
    if len(text) <= size:
        return [text]
    words = text.split()
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if sum(len(x) + 1 for x in cur) >= size:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [text]


# ── search ─────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = 9


@app.post("/api/search")
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(400, "query is empty")

    client  = _openrouter_client()
    reducer = _get_reducer()

    # Embed query (chunked for long inputs)
    chunks = _chunk_text(req.query, size=200) if len(req.query) > 500 else [req.query]
    chunk_vecs = embed_texts(chunks, client)          # (N_chunks, 1536)
    centroid    = chunk_vecs.mean(axis=0)             # (1536,)
    q3d         = reducer.transform([centroid])[0]    # (3,)

    conn = get_conn()
    cur  = conn.cursor()

    # ── find nearest domain ──────────────────────────────────────────────────
    cur.execute("""
        SELECT id, name, centroid, density,
               ST_Distance(centroid_3d, ST_GeomFromEWKT(%s)) AS spatial_dist
        FROM skill_selector.domains
        WHERE centroid IS NOT NULL
    """, (f"SRID=0;POINTZ({q3d[0]} {q3d[1]} {q3d[2]})",))

    domains = cur.fetchall()
    if not domains:
        conn.close()
        return JSONResponse({"results": [], "query_point_3d": q3d.tolist(), "domain": None})

    # Pick best domain by raw cosine to centroid — density normalisation
    # inflates scores for tight micro-clusters and breaks routing.
    best_domain = None
    best_cos    = -999.0
    for d in domains:
        cos = _cosine(centroid, _parse_vec(d["centroid"]))
        if cos > best_cos:
            best_cos    = cos
            best_domain = d
    best_score = best_cos  # keep for response

    # ── retrieve top candidates ───────────────────────────────────────────────
    DOMAIN_CONFIDENCE_THRESHOLD = 0.35
    use_domain = best_domain and best_cos >= DOMAIN_CONFIDENCE_THRESHOLD
    domain_name = best_domain["name"] if best_domain else None
    domain_3d   = f"SRID=0;POINTZ({q3d[0]} {q3d[1]} {q3d[2]})"

    if use_domain:
        cur.execute("""
            SELECT
                s.id, s.name, s.url, s.source_repo, s.category,
                s.description, s.size, s.char_count, s.embed_tier,
                ST_X(s.point_3d) AS px, ST_Y(s.point_3d) AS py, ST_Z(s.point_3d) AS pz,
                ST_Distance(s.point_3d, ST_GeomFromEWKT(%s)) AS spatial_dist,
                1 - (s.embedding <=> %s::vector) AS cosine_score,
                ST_X(sd.point_3d_local) AS lx,
                ST_Y(sd.point_3d_local) AS ly,
                ST_Z(sd.point_3d_local) AS lz
            FROM skill_selector.skills s
            JOIN skill_selector.skill_domains sd ON sd.skill_id = s.id
            JOIN skill_selector.domains d ON d.id = sd.domain_id
            WHERE d.name = %s
            ORDER BY s.embedding <=> %s::vector
            LIMIT 50
        """, (domain_3d, centroid.tolist(), domain_name, centroid.tolist()))
    else:
        # Global HNSW search — no domain filter
        domain_name = None
        cur.execute("""
            SELECT
                s.id, s.name, s.url, s.source_repo, s.category,
                s.description, s.size, s.char_count, s.embed_tier,
                ST_X(s.point_3d) AS px, ST_Y(s.point_3d) AS py, ST_Z(s.point_3d) AS pz,
                ST_Distance(s.point_3d, ST_GeomFromEWKT(%s)) AS spatial_dist,
                1 - (s.embedding <=> %s::vector) AS cosine_score,
                NULL AS lx, NULL AS ly, NULL AS lz
            FROM skill_selector.skills s
            ORDER BY s.embedding <=> %s::vector
            LIMIT 50
        """, (domain_3d, centroid.tolist(), centroid.tolist()))

    rows = cur.fetchall()
    conn.close()

    # Re-rank: 70% cosine + 30% inverted spatial
    results = []
    for r in rows:
        cos   = float(r["cosine_score"] or 0)
        sdist = float(r["spatial_dist"] or 999)
        score = 0.7 * cos + 0.3 / (1.0 + sdist)
        results.append({**dict(r), "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[: req.top_k]

    # Chunk points (one per chunk + centroid)
    chunk_points = []
    if len(chunks) > 1:
        chunk_3ds = reducer.transform(chunk_vecs)
        for i, c3 in enumerate(chunk_3ds):
            chunk_points.append({
                "x": float(c3[0]), "y": float(c3[1]), "z": float(c3[2]),
                "is_centroid": False, "chunk_index": i,
            })
    chunk_points.append({
        "x": float(q3d[0]), "y": float(q3d[1]), "z": float(q3d[2]),
        "is_centroid": True,
    })

    return JSONResponse({
        "results": [
            {
                "name":         r["name"],
                "url":          r["url"],
                "source_repo":  r["source_repo"],
                "category":     r["category"],
                "description":  r["description"],
                "size":         r["size"],
                "score":        round(r["score"], 4),
                "cosine_score": round(float(r["cosine_score"] or 0), 4),
                "spatial_dist": round(float(r["spatial_dist"] or 0), 4),
                "embed_tier":   r["embed_tier"],
                "point_3d":     {"x": r["px"], "y": r["py"], "z": r["pz"]},
                "point_3d_local": {"x": r["lx"], "y": r["ly"], "z": r["lz"]},
            }
            for r in top
        ],
        "domain":        domain_name,
        "domain_score":  round(best_score, 4),
        "query_points":  chunk_points,
        "domain_centroid_3d": {
            "x": float(best_domain.get("centroid_3d_x") or q3d[0]),
            "y": float(best_domain.get("centroid_3d_y") or q3d[1]),
            "z": float(best_domain.get("centroid_3d_z") or q3d[2]),
        },
    })


# ── compare: spatial vs pure-semantic ─────────────────────────────────────────

@app.post("/api/compare")
def compare(req: SearchRequest):
    """Return top-3 spatial AND top-3 pure-semantic results for the same query."""
    if not req.query.strip():
        raise HTTPException(400, "query is empty")

    client  = _openrouter_client()
    reducer = _get_reducer()

    chunks     = _chunk_text(req.query, size=200) if len(req.query) > 500 else [req.query]
    chunk_vecs = embed_texts(chunks, client)
    centroid   = chunk_vecs.mean(axis=0)
    q3d        = reducer.transform([centroid])[0]
    domain_3d  = f"SRID=0;POINTZ({q3d[0]} {q3d[1]} {q3d[2]})"

    conn = get_conn()
    cur  = conn.cursor()

    # ── Pure semantic: global HNSW, cosine only ───────────────────────────────
    cur.execute("""
        SELECT name, url, description, source_repo, category, size, embed_tier, tags,
               ST_X(point_3d) AS px, ST_Y(point_3d) AS py, ST_Z(point_3d) AS pz,
               1 - (embedding <=> %s::vector) AS cosine_score
        FROM skill_selector.skills
        ORDER BY embedding <=> %s::vector
        LIMIT 3
    """, (centroid.tolist(), centroid.tolist()))
    sem_rows = cur.fetchall()

    # ── Spatial: pure PostGIS KNN on point_3d — no embedding lookup ──────────
    cur.execute("""
        SELECT s.name, s.url, s.description, s.source_repo, s.category, s.size, s.embed_tier, s.tags,
               ST_X(s.point_3d) AS px, ST_Y(s.point_3d) AS py, ST_Z(s.point_3d) AS pz,
               ST_Distance(s.point_3d, ST_GeomFromEWKT(%s)) AS spatial_dist
        FROM skill_selector.skills s
        WHERE s.point_3d IS NOT NULL
        ORDER BY s.point_3d <-> ST_GeomFromEWKT(%s)
        LIMIT 3
    """, (domain_3d, domain_3d))
    spa_rows = cur.fetchall()

    # Nearest domain by spatial distance (label only)
    cur.execute("""
        SELECT name
        FROM skill_selector.domains
        WHERE centroid_3d IS NOT NULL
        ORDER BY centroid_3d <-> ST_GeomFromEWKT(%s)
        LIMIT 1
    """, (domain_3d,))
    dom_row     = cur.fetchone()
    domain_name = dom_row["name"] if dom_row else None
    domain_centroid = {"x": float(q3d[0]), "y": float(q3d[1]), "z": float(q3d[2])}

    conn.close()

    chunk_points = []
    if len(chunks) > 1:
        for c3 in reducer.transform(chunk_vecs):
            chunk_points.append({"x": float(c3[0]), "y": float(c3[1]), "z": float(c3[2]), "is_centroid": False})
    chunk_points.append({"x": float(q3d[0]), "y": float(q3d[1]), "z": float(q3d[2]), "is_centroid": True})

    def _fmt(r, score):
        return {
            "name":        r["name"],
            "url":         r["url"],
            "description": r["description"],
            "source_repo": r["source_repo"],
            "category":    r["category"],
            "size":        r["size"],
            "tags":        r.get("tags") or [],
            "score":       round(float(score), 4),
            "cosine_score": round(float(r.get("cosine_score") or 0), 4),
            "spatial_dist": round(float(r.get("spatial_dist") or 0), 4),
            "point_3d":    {"x": r["px"], "y": r["py"], "z": r["pz"]},
        }

    return JSONResponse({
        "semantic":          [_fmt(r, r["cosine_score"]) for r in sem_rows],
        "spatial":           [_fmt(r, 1.0 / (1.0 + float(r["spatial_dist"] or 999))) for r in spa_rows],
        "query_points":      chunk_points,
        "domain":            domain_name,
        "domain_centroid_3d": domain_centroid,
    })


# ── skills list / detail ───────────────────────────────────────────────────────

@app.get("/api/skills")
def list_skills(
    category: str = Query(None),
    size:     str = Query(None),
    page:     int = Query(1, ge=1),
    per_page: int = Query(24, ge=1, le=100),
):
    conn = get_conn()
    cur  = conn.cursor()
    where, params = ["1=1"], []
    if category:
        where.append("category = %s"); params.append(category)
    if size:
        where.append("size = %s"); params.append(size.upper())
    w = " AND ".join(where)
    cur.execute(f"SELECT COUNT(*) AS n FROM skill_selector.skills WHERE {w}", params)
    total = cur.fetchone()["n"]
    offset = (page - 1) * per_page
    cur.execute(f"""
        SELECT name, url, source_repo, category, description, size, embed_tier, user_score
        FROM skill_selector.skills WHERE {w}
        ORDER BY name LIMIT %s OFFSET %s
    """, params + [per_page, offset])
    rows = cur.fetchall()
    conn.close()
    return JSONResponse({"total": total, "page": page, "per_page": per_page, "skills": [dict(r) for r in rows]})


@app.get("/api/skills/{name:path}")
def get_skill(name: str):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        SELECT name, url, source_repo, category, description, body,
               author, size, char_count, embed_text, embed_tier,
               user_score, created_at, updated_at
        FROM skill_selector.skills WHERE name = %s LIMIT 1
    """, (name,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, f"skill '{name}' not found")
    d = {k: (v.isoformat() if hasattr(v, 'isoformat') else v) for k, v in dict(row).items()}
    return JSONResponse(d)


# ── pointcloud ─────────────────────────────────────────────────────────────────

@app.get("/api/pointcloud")
def pointcloud():
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        SELECT s.name, s.category, s.size,
               ST_X(s.point_3d) AS x, ST_Y(s.point_3d) AS y, ST_Z(s.point_3d) AS z
        FROM skill_selector.skills s
        WHERE s.point_3d IS NOT NULL
    """)
    skills = [dict(r) for r in cur.fetchall()]

    cur.execute("""
        SELECT name, skill_count, density,
               ST_X(centroid_3d) AS cx, ST_Y(centroid_3d) AS cy, ST_Z(centroid_3d) AS cz
        FROM skill_selector.domains
        WHERE centroid_3d IS NOT NULL
    """)
    domains = [dict(r) for r in cur.fetchall()]

    # r60: 60th-pct distance from centroid per domain
    cur.execute("""
        SELECT d.name AS domain,
               percentile_cont(0.6) WITHIN GROUP (
                   ORDER BY ST_Distance(s.point_3d, d.centroid_3d)
               ) AS r60
        FROM skill_selector.skills s
        JOIN skill_selector.domains d ON d.name = s.category
        WHERE s.point_3d IS NOT NULL AND d.centroid_3d IS NOT NULL
        GROUP BY d.name
    """)
    radii = {r["domain"]: r["r60"] for r in cur.fetchall()}
    conn.close()

    for d in domains:
        d["r60"] = radii.get(d["name"], 1.0)

    return JSONResponse({"skills": skills, "domains": domains})


@app.get("/api/pointcloud/domain/{name}")
def pointcloud_domain(name: str):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        SELECT s.name, s.size,
               ST_X(sd.point_3d_local) AS x,
               ST_Y(sd.point_3d_local) AS y,
               ST_Z(sd.point_3d_local) AS z
        FROM skill_selector.skills s
        JOIN skill_selector.skill_domains sd ON sd.skill_id = s.id
        JOIN skill_selector.domains d ON d.id = sd.domain_id
        WHERE d.name = %s AND sd.point_3d_local IS NOT NULL
    """, (name,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    if not rows:
        raise HTTPException(404, f"domain '{name}' not found or has no local projections")
    return JSONResponse({"domain": name, "skills": rows})


@app.get("/api/categories")
def categories():
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        SELECT category, COUNT(*) AS n
        FROM skill_selector.skills
        GROUP BY category ORDER BY n DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return JSONResponse([dict(r) for r in rows])


# ── root ───────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8200))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("server:app", host=host, port=port, reload=True)
