"""Migrations du store SQLite.

Le schéma est dérivé dynamiquement de `RuleRecord` dans `RuleStore._init_schema`,
ce qui couvre l'ajout de colonnes pour la V1. Ce module centralise les
migrations explicites futures (renommages, backfills) et la version de schéma.
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 1


def current_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def set_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(f"PRAGMA user_version = {int(version)}")
    conn.commit()


def migrate(conn: sqlite3.Connection) -> int:
    """Applique les migrations en attente. Renvoie la version finale.

    V1 : aucune migration destructive ; on aligne juste `user_version`.
    """
    if current_version(conn) < SCHEMA_VERSION:
        set_version(conn, SCHEMA_VERSION)
    return SCHEMA_VERSION
