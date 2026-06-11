"""PAGEPRINT Serializers — sérialisation JSON de INPUT_DATA."""

from __future__ import annotations

import json
import os


def _default(value):
    try:
        return str(value)
    except Exception:
        return None


def to_json(input_data: dict, *, indent: int | None = 2,
            exclude_keys: tuple = ()) -> str:
    if exclude_keys:
        input_data = {k: v for k, v in input_data.items() if k not in set(exclude_keys)}
    return json.dumps(input_data, ensure_ascii=False, indent=indent, default=_default)


def save_input_data(input_data: dict, path: str, *, indent: int | None = 2,
                    exclude_keys: tuple = ()) -> str:
    """Sauvegarde INPUT_DATA.

    `exclude_keys` permet d'omettre des vues dérivées volumineuses
    (ex: "compatibility", qui n'est pas la source de vérité) dans les
    exports d'étude/audit.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(to_json(input_data, indent=indent, exclude_keys=exclude_keys))
    return path


def load_input_data(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
