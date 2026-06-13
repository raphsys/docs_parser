from __future__ import annotations

from dataclasses import asdict, dataclass, field
from uuid import uuid4


@dataclass
class TranslationRequest:
    text: str
    source_lang: str = "auto"
    target_lang: str = "fr"
    context: dict = field(default_factory=dict)
    protected_text: str | None = None
    protected_tokens: list[str] = field(default_factory=list)
    translation_unit_id: str | None = None
    role: str | None = None
    object_type: str | None = None
    semantic_kind: str | None = None
    constraints: dict = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: f"trq_{uuid4().hex[:12]}")

    def to_dict(self) -> dict:
        return asdict(self)
