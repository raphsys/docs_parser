"""Explicit render operations (directive Lot E). The backend executes these;
it does not improvise."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BackgroundOp:
    op_type: str
    mode: str
    path: str | None
    page_rect: list | None = None

    def to_dict(self):
        return self.__dict__.copy()


@dataclass
class PatchOp:
    op_type: str
    unit_id: str
    bbox: list
    method: str
    color: str | None = None
    protected_overlap_ratio: float = 0.0
    status: str = "ok"

    def to_dict(self):
        return self.__dict__.copy()


@dataclass
class TextOp:
    op_type: str
    unit_id: str
    text: str
    bbox: list
    resolved_style: dict
    renderer: str
    align: str = "left"
    z_index: int = 4

    def to_dict(self):
        return self.__dict__.copy()


@dataclass
class PreservationOp:
    op_type: str
    unit_id: str
    bbox: list
    method: str = "copy_region_from_source"
    z_index: int = 5

    def to_dict(self):
        return self.__dict__.copy()


@dataclass
class Finding:
    severity: str
    code: str
    message: str = ""
    unit_id: str | None = None
    bbox: list | None = None

    def to_dict(self):
        return self.__dict__.copy()
