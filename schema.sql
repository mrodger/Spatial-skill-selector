-- Skill Selector schema
-- Run: psql $DATABASE_URL -f schema.sql

CREATE SCHEMA IF NOT EXISTS skill_selector;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS postgis;

-- ── skills ────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS skill_selector.skills (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    url             TEXT NOT NULL,
    source_repo     TEXT NOT NULL,          -- e.g. "anthropics/skills"
    category        TEXT NOT NULL DEFAULT 'Uncategorised',
    description     TEXT NOT NULL,
    body            TEXT,
    author          TEXT,
    user_score      REAL,
    size            TEXT NOT NULL CHECK (size IN ('S', 'M', 'L')),
    char_count      INTEGER NOT NULL,
    embed_text      TEXT NOT NULL,          -- exact text embedded (reproducible)
    embed_tier      SMALLINT NOT NULL,      -- 1=when-to-use 2=capabilities 3=fallback
    domain_inferred BOOLEAN NOT NULL DEFAULT false,
    embedding       vector(1536) NOT NULL,
    point_3d        geometry(PointZ, 0),    -- global UMAP projection
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE (name, source_repo)
);

CREATE INDEX IF NOT EXISTS idx_skills_embedding
    ON skill_selector.skills USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_skills_point_3d
    ON skill_selector.skills USING gist (point_3d);

CREATE INDEX IF NOT EXISTS idx_skills_category
    ON skill_selector.skills (category);

-- ── domains ───────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS skill_selector.domains (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    centroid    vector(1536),
    centroid_3d geometry(PointZ, 0),
    density     REAL,                       -- mean pairwise cosine dist within cluster
    skill_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_domains_centroid_3d
    ON skill_selector.domains USING gist (centroid_3d);

-- ── skill_domains (junction + local projection) ───────────────────────────────

CREATE TABLE IF NOT EXISTS skill_selector.skill_domains (
    skill_id        INTEGER NOT NULL REFERENCES skill_selector.skills (id) ON DELETE CASCADE,
    domain_id       INTEGER NOT NULL REFERENCES skill_selector.domains (id) ON DELETE CASCADE,
    point_3d_local  geometry(PointZ, 0),    -- per-domain local UMAP projection
    PRIMARY KEY (skill_id, domain_id)
);

CREATE INDEX IF NOT EXISTS idx_skill_domains_local
    ON skill_selector.skill_domains USING gist (point_3d_local);
