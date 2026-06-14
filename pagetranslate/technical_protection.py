"""Protect technical tokens in code/table contexts (directive Lot 9).

In technical cells/code, things like None/True/False, Conv2D, tensor shapes,
SQL keywords, file paths and function calls must not be translated/altered.
"""

from __future__ import annotations

import re

TECHNICAL_ROLES = {
    "table_body_cell", "table_header_cell", "table_numeric_cell",
    "code_line", "code_block", "command_name", "path", "file_name", "formula_expression",
}

_PATTERNS = [
    re.compile(r"\b(?:Conv2D|MaxPooling2D|AveragePooling2D|InputLayer|Dense|Flatten|Dropout|"
               r"BatchNormalization|LSTM|GRU|Embedding|Softmax|ReLU|Sigmoid|Concatenate)\b"),
    re.compile(r"\(\s*None(?:\s*,\s*\d+)+\s*\)"),                 # tensor shapes (None, 26, 26, 32)
    re.compile(r"\b(?:None|True|False|NaN|null)\b"),
    re.compile(r"\b(?:SELECT|FROM|WHERE|INSERT|UPDATE|DELETE|COMMIT|ROLLBACK|JOIN|GROUP BY|"
               r"ORDER BY|CREATE TABLE|DROP TABLE|START TRANSACTION)\b"),
    re.compile(r"(?:[A-Za-z]:\\[^\s]+|/[\w.][\w./-]+|\*\.\w+)"),  # paths / file globs
    re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\([^)]*\)"),            # function calls f(...) / ST_AsText()
    re.compile(r"\b[a-z]+_[a-z_]+\b"),                            # snake_case identifiers (pg_restore)
]


def technical_tokens(text: str) -> list[str]:
    text = str(text or "")
    out: list[str] = []
    for pat in _PATTERNS:
        out.extend(m.group(0) for m in pat.finditer(text))
    # keep order, unique, only tokens actually present
    return list(dict.fromkeys(t for t in out if t and t in text))


def is_technical_role(role: str | None, object_type: str | None = None) -> bool:
    if str(role or "") in TECHNICAL_ROLES:
        return True
    return str(object_type or "").lower() in {"code", "code_block", "table_cell", "formula_expression", "path"}
