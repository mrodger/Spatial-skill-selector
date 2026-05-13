"""GitHub crawler — git clone (fast) + API fallback + link-list."""
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Iterator

import requests
import yaml


API = "https://api.github.com"
SKILL_FILE = "SKILL.md"


def _headers() -> dict:
    tok = os.environ.get("GITHUB_TOKEN", "")
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _get(url: str) -> requests.Response:
    for attempt in range(4):
        try:
            r = requests.get(url, headers=_headers(), timeout=30)
        except Exception as e:
            wait = 10 * (attempt + 1)
            print(f"  connection error ({e}), retry in {wait}s")
            time.sleep(wait)
            continue
        remaining = int(r.headers.get("X-RateLimit-Remaining", 999))
        if remaining < 10:
            reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - time.time() + 2, 5)
            print(f"  rate limit low ({remaining}), sleeping {wait:.0f}s")
            time.sleep(wait)
        if r.status_code == 200:
            return r
        if r.status_code == 404:
            return r
        if r.status_code in (403, 429):
            time.sleep(30 * (attempt + 1))
        else:
            time.sleep(2)
    # Return a fake 503 response on exhausted retries
    r = requests.Response()
    r.status_code = 503
    return r


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Split YAML frontmatter from body. Returns (meta, body)."""
    if not content.startswith("---"):
        return {}, content
    end = content.find("\n---", 3)
    if end == -1:
        return {}, content
    try:
        meta = yaml.safe_load(content[3:end]) or {}
    except yaml.YAMLError:
        meta = {}
    body = content[end + 4:].lstrip("\n")
    return meta, body


def _decode_content(item: dict) -> str:
    """Decode base64 file content from GitHub API response."""
    import base64
    raw = item.get("content", "")
    return base64.b64decode(raw.replace("\n", "")).decode("utf-8", errors="replace")


def _clone_repo(owner: str, repo: str, branch: str, target_dir: str) -> bool:
    """Shallow-clone a repo into target_dir. Returns True on success."""
    tok = os.environ.get("GITHUB_TOKEN", "")
    if tok:
        url = f"https://{tok}@github.com/{owner}/{repo}.git"
    else:
        url = f"https://github.com/{owner}/{repo}.git"
    result = subprocess.run(
        ["git", "clone", "--depth=1", "--branch", branch, url, target_dir],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"  clone failed: {result.stderr.strip()[:120]}")
        return False
    return True


def crawl_hosted(
    owner: str,
    repo: str,
    branch: str = "main",
    skip_prefixes: list[str] | None = None,
    category_map: dict[str, str] | None = None,
) -> Iterator[dict]:
    """
    Yield one dict per SKILL.md found in the repo.
    Uses git clone --depth=1 for speed instead of per-file API calls.
    """
    skip_prefixes = skip_prefixes or []
    category_map = category_map or {}

    tmpdir = tempfile.mkdtemp(prefix=f"skill_{owner}_{repo}_")
    try:
        print(f"  cloning {owner}/{repo} @ {branch} → {tmpdir}")
        if not _clone_repo(owner, repo, branch, tmpdir):
            return

        found = 0
        for root, dirs, files in os.walk(tmpdir):
            # Skip .git
            dirs[:] = [d for d in dirs if d != ".git"]
            if SKILL_FILE not in files:
                continue

            filepath = os.path.join(root, SKILL_FILE)
            # Relative path from repo root
            rel_path = os.path.relpath(filepath, tmpdir).replace(os.sep, "/")

            if any(rel_path.startswith(p) for p in skip_prefixes):
                continue

            try:
                content = open(filepath, encoding="utf-8", errors="replace").read()
                content = content.replace("\x00", "")
            except OSError:
                continue

            meta, body = _parse_frontmatter(content)

            # Name: frontmatter → parent dir name
            parent_dir = os.path.basename(os.path.dirname(filepath))
            name = meta.get("name") or parent_dir

            description = meta.get("description", "")
            if not description:
                for line in body.splitlines():
                    line = line.strip().lstrip("#").strip()
                    if line:
                        description = line[:200]
                        break
            if not description:
                continue

            # Category: category_map prefix match → frontmatter → Uncategorised
            category = "Uncategorised"
            for prefix, cat in category_map.items():
                if rel_path.startswith(prefix):
                    category = cat
                    break
            if category == "Uncategorised":
                category = str(meta.get("category", "Uncategorised"))

            char_count = len(content)
            size = "S" if char_count < 3000 else ("M" if char_count < 10000 else "L")
            skill_dir = rel_path.rsplit("/", 1)[0] if "/" in rel_path else rel_path

            found += 1
            yield {
                "name":        str(name).strip(),
                "url":         f"https://github.com/{owner}/{repo}/tree/{branch}/{skill_dir}",
                "source_repo": f"{owner}/{repo}",
                "description": str(description).strip(),
                "body":        body,
                "author":      str(meta.get("author", "")),
                "category":    category,
                "char_count":  char_count,
                "size":        size,
                "tags":        meta.get("tags") or meta.get("capabilities") or [],
            }

        print(f"  {found} skills found in {owner}/{repo}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


_GH_LINK = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")


def crawl_linklist(
    owner: str,
    repo: str,
    branch: str = "main",
    already_seen: set[str] | None = None,
) -> Iterator[dict]:
    """
    Parse README.md of a link-list repo and crawl each linked GitHub repo
    for SKILL.md files. Skips repos already seen.
    """
    already_seen = already_seen or set()
    # fetch README directly via API (single call)
    r = _get(f"{API}/repos/{owner}/{repo}/contents/README.md?ref={branch}")
    if r.status_code != 200:
        return
    readme = _decode_content(r.json())

    seen_in_list: set[str] = set()
    for m in _GH_LINK.finditer(readme):
        slug = m.group(1).rstrip("/.")
        if slug in seen_in_list or slug in already_seen:
            continue
        seen_in_list.add(slug)
        link_owner, link_repo = slug.split("/", 1)
        print(f"  link-list → checking {slug}")
        yield from crawl_hosted(link_owner, link_repo)


# ── repo registry ──────────────────────────────────────────────────────────────

REPOS_HOSTED = [
    {
        "owner": "anthropics", "repo": "skills", "branch": "main",
        "category_map": {
            "skills/docx": "Documents", "skills/pdf": "Documents",
            "skills/pptx": "Documents", "skills/xlsx": "Documents",
            "skills/algorithmic-art": "Design & Creative",
            "skills/canvas-design": "Design & Creative",
            "skills/slack-gif-creator": "Design & Creative",
            "skills/brand-guidelines": "Communication",
            "skills/internal-comms": "Communication",
            "skills/frontend-design": "Development",
            "skills/web-artifacts-builder": "Development",
            "skills/webapp-testing": "Development",
            "skills/mcp-builder": "Development",
            "skills/skill-creator": "Skill Creation",
            "skills/theme-factory": "Design & Creative",
            "skills/claude-api": "Development",
            "skills/doc-coauthoring": "Documents",
        },
    },
    {
        "owner": "obra", "repo": "superpowers", "branch": "main",
        "category_map": {"skills/": "Development Workflow"},
    },
    {
        "owner": "ComposioHQ", "repo": "awesome-claude-skills", "branch": "master",
        "skip_prefixes": ["composio-skills/", "template-skill/"],
        "category_map": {
            "document-skills/": "Documents",
            "mcp-builder/": "Development",
            "webapp-testing/": "Development",
            "connect": "Integrations",
        },
    },
    {
        "owner": "sickn33", "repo": "antigravity-awesome-skills", "branch": "main",
        "category_map": {"skills/": "Community"},
    },
    {
        "owner": "alirezarezvani", "repo": "claude-skills", "branch": "main",
        "category_map": {},
    },
    {
        "owner": "obra", "repo": "superpowers-skills", "branch": "main",
        "category_map": {},
    },
]

REPOS_LINKLIST = [
    {"owner": "travisvn",      "repo": "awesome-claude-skills", "branch": "main"},
    {"owner": "BehiSecc",      "repo": "awesome-claude-skills", "branch": "main"},
    {"owner": "karanb192",     "repo": "awesome-claude-skills", "branch": "main"},
]

SEED_REPOS = ["anthropics/skills", "obra/superpowers"]
