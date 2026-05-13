"""PostGIS connection helpers."""
import os
from pathlib import Path

import psycopg2
import psycopg2.extras


def _load_secrets():
    p = Path.home() / ".secrets.env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_secrets()


def get_conn():
    conn = psycopg2.connect(
        host=os.environ["POSTGIS_HOST"],
        port=int(os.environ.get("POSTGIS_PORT", 5432)),
        dbname=os.environ["POSTGIS_DATABASE"],
        user=os.environ["POSTGIS_USER"],
        password=os.environ.get("POSTGIS_PASSWORD", "password"),
        cursor_factory=psycopg2.extras.RealDictCursor,
    )
    conn.set_client_encoding("UTF8")
    return conn
