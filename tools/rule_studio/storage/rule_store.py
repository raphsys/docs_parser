"""Store SQLite des `RuleRecord`.

Le store est la source de vérité de l'inventaire en cours. Il fusionne les
règles re-scannées avec celles déjà présentes en préservant les champs édités
par l'humain (décisions, objectif, compréhension, statut, risque).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import fields
from pathlib import Path
from typing import Any

from ..core.models import RuleRecord

# Champs édités par l'humain : préservés lors d'un re-scan.
_HUMAN_FIELDS = (
    "objective",
    "understanding",
    "keep_decision",
    "modification_decision",
    "proposed_modification",
    "status",
    "risk_level",
    "managed",
    "validation_status",
    "tests_linked",
)

_LIST_FIELDS = ("secondary_units", "tests_linked")
_JSON_FIELDS = ("dsl", "target")

# Tous les champs booléens de RuleRecord (anciens + décisions + agents) : stockés
# "0"/"1" et relus en vrais bool. Dérivé de l'annotation de type pour ne jamais
# oublier un nouveau booléen ajouté au modèle.
from ..core.models import BOOL_DECISION_FIELDS  # noqa: E402

_BOOL_FIELDS = tuple(dict.fromkeys(
    ("managed", "editable", "deletable", "simulable", "ai_interpreted")
    + BOOL_DECISION_FIELDS
))


class RuleStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False : la connexion est partagée entre les threads
        # du pool Streamlit (`@st.cache_resource`). On sérialise tous les accès
        # via un verrou pour rester correct malgré ce partage.
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    # -- schéma --------------------------------------------------------------
    def _init_schema(self) -> None:
        cols = [f.name for f in fields(RuleRecord)]
        col_defs = ",\n".join(f'"{c}" TEXT' for c in cols if c != "id")
        with self._lock:
            self.conn.execute(
                f'CREATE TABLE IF NOT EXISTS rules (\n'
                f'  "id" TEXT PRIMARY KEY,\n  {col_defs}\n)'
            )
            # Migration légère : ajoute les colonnes manquantes sur une base
            # existante (nouveaux champs naturels / booléens / agents / lifecycle).
            existing = {r[1] for r in self.conn.execute('PRAGMA table_info(rules)')}
            for c in cols:
                if c != "id" and c not in existing:
                    self.conn.execute(f'ALTER TABLE rules ADD COLUMN "{c}" TEXT')
            self.conn.commit()

    # -- (dé)sérialisation ---------------------------------------------------
    @staticmethod
    def _to_db(rec: RuleRecord) -> dict[str, Any]:
        d = rec.to_dict()
        for k in _LIST_FIELDS:
            d[k] = json.dumps(d.get(k) or [], ensure_ascii=False)
        for k in _JSON_FIELDS:
            d[k] = json.dumps(d.get(k) or {}, ensure_ascii=False)
        for k in _BOOL_FIELDS:
            d[k] = "1" if d.get(k) else "0"
        return d

    @staticmethod
    def _from_db(row: sqlite3.Row) -> RuleRecord:
        d = dict(row)
        for k in _LIST_FIELDS:
            try:
                d[k] = json.loads(d.get(k) or "[]")
            except (json.JSONDecodeError, TypeError):
                d[k] = []
        for k in _JSON_FIELDS:
            try:
                d[k] = json.loads(d.get(k) or "{}")
            except (json.JSONDecodeError, TypeError):
                d[k] = {}
        for k in _BOOL_FIELDS:
            d[k] = str(d.get(k)) in {"1", "True", "true"}
        if d.get("confidence") not in (None, ""):
            try:
                d["confidence"] = float(d["confidence"])
            except (ValueError, TypeError):
                d["confidence"] = 0.0
        for k in ("line_start", "line_end"):
            v = d.get(k)
            d[k] = int(v) if str(v).isdigit() else None
        return RuleRecord.from_dict(d)

    # -- API ----------------------------------------------------------------
    def upsert(self, rec: RuleRecord) -> None:
        d = self._to_db(rec)
        cols = list(d.keys())
        placeholders = ",".join("?" for _ in cols)
        col_list = ",".join(f'"{c}"' for c in cols)
        updates = ",".join(f'"{c}"=excluded."{c}"' for c in cols if c != "id")
        with self._lock:
            self.conn.execute(
                f"INSERT INTO rules ({col_list}) VALUES ({placeholders}) "
                f"ON CONFLICT(id) DO UPDATE SET {updates}",
                [d[c] for c in cols],
            )
            self.conn.commit()

    def get(self, rule_id: str) -> RuleRecord | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM rules WHERE id=?", (rule_id,)
            ).fetchone()
        return self._from_db(row) if row else None

    def all(self) -> list[RuleRecord]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM rules ORDER BY pipeline_unit, id"
            ).fetchall()
        return [self._from_db(r) for r in rows]

    def delete(self, rule_id: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM rules WHERE id=?", (rule_id,))
            self.conn.commit()

    def count(self) -> int:
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) FROM rules").fetchone()[0]

    def merge_scan(self, scanned: list[RuleRecord]) -> dict[str, int]:
        """Fusionne un lot scanné en préservant les éditions humaines.

        Renvoie un compte {added, updated, preserved_edits}.
        """
        stats = {"added": 0, "updated": 0, "preserved": 0}
        for rec in scanned:
            existing = self.get(rec.id)
            if existing is None:
                self.upsert(rec)
                stats["added"] += 1
                continue
            # Préserve les champs édités par l'humain.
            edited = False
            for fld in _HUMAN_FIELDS:
                human_val = getattr(existing, fld)
                default_val = getattr(rec, fld)
                if human_val and human_val != default_val:
                    setattr(rec, fld, human_val)
                    edited = True
            rec.created_at = existing.created_at or rec.created_at
            rec.touch()
            self.upsert(rec)
            stats["updated"] += 1
            if edited:
                stats["preserved"] += 1
        return stats

    def close(self) -> None:
        self.conn.close()
