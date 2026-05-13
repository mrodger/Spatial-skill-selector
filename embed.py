"""Tiered embed-text extraction and batch embedding."""
import re
from typing import Optional

import numpy as np


# ── text cleaning ──────────────────────────────────────────────────────────────

_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_HTML_TAG   = re.compile(r"<[^>]+>")
_MD_TABLE   = re.compile(r"(\|.+\|[\r\n]+)+")
_BOILERPLATE = re.compile(
    r"(## Limitations.*|## Constraints.*|## Additional Resources.*"
    r"|## Troubleshooting.*|## Examples.*"
    r"|When to Use\nThis skill is applicable.*)",
    re.DOTALL | re.IGNORECASE,
)


def _clean(text: str) -> str:
    text = _CODE_BLOCK.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _MD_TABLE.sub(" ", text)
    text = _BOILERPLATE.sub("", text)
    text = re.sub(r"\s{3,}", " ", text)
    return text.strip()


# ── tier extraction ────────────────────────────────────────────────────────────

def _extract_when_to_use(body: str) -> Optional[str]:
    """Extract 'User mentions or implies: X' bullets from ## When to Use."""
    m = re.search(r"##\s+When to Use\b(.*?)(?=\n##|\Z)", body, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    section = m.group(1)
    triggers = re.findall(
        r"(?:User mentions or implies|When user mentions?|Use when)[:\s]+(.+)",
        section,
        re.IGNORECASE,
    )
    if not triggers:
        return None
    return "; ".join(t.strip().strip("-").strip() for t in triggers if t.strip())


def _extract_capabilities(body: str, frontmatter_tags: Optional[list]) -> Optional[str]:
    """Extract ## Capabilities list or frontmatter tags."""
    if frontmatter_tags:
        return "; ".join(str(t) for t in frontmatter_tags)
    m = re.search(r"##\s+Capabilities\b(.*?)(?=\n##|\Z)", body, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    items = re.findall(r"[-*]\s+(.+)", m.group(1))
    return "; ".join(i.strip() for i in items) if items else None


def build_embed_text(description: str, body: str, frontmatter_tags: Optional[list] = None):
    """
    Returns (embed_text, tier).
    Tier 1: When-to-use triggers
    Tier 2: Capabilities / tags
    Tier 3: description + first 300 chars of cleaned body
    """
    desc = description.strip()

    triggers = _extract_when_to_use(body)
    if triggers:
        return f"{desc}\n{triggers}", 1

    caps = _extract_capabilities(body, frontmatter_tags)
    if caps:
        return f"{desc}\n{caps}", 2

    snippet = _clean(body)[:300].strip()
    return f"{desc}\n{snippet}" if snippet else desc, 3


# ── embedding ──────────────────────────────────────────────────────────────────

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE  = 100


def embed_texts(texts: list[str], client) -> np.ndarray:
    """Embed a list of strings in batches. Returns float32 ndarray (N, 1536)."""
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(e.embedding for e in resp.data)
    return np.array(vectors, dtype=np.float32)
