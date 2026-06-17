"""BackgroundContract — quel fond utiliser + risque de fuite du texte source.

clean_background (moderne ou legacy master) > source_background (debug seulement)
> blank_degraded. En mode publication, un fond non nettoyé est bloquant.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class BackgroundContract:
    clean_background_path: str | None = None
    source_image_path: str | None = None
    background_mode: str = "blank_degraded"   # clean_background | source_background | blank_degraded
    inpaint_masks: list = field(default_factory=list)
    text_removed: bool = False
    clean_background_verified: bool = False
    source_text_leak_risk: str = "high"       # low | medium | high | critical
    publication_allowed: bool = False
    findings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def path(self) -> str | None:
        """Fond générique (debug). NE PAS utiliser pour le rendu publication :
        passer par render_path(mode) qui interdit le fallback source."""
        return self.clean_background_path or self.source_image_path

    def render_path(self, mode: str = "debug") -> str | None:
        """Fond autorisé pour le rendu selon le mode.
        publication : UNIQUEMENT clean_background_path (jamais la source).
        debug/review : clean si dispo, sinon source (overlay debug toléré)."""
        if mode == "publication":
            return self.clean_background_path
        return self.clean_background_path or self.source_image_path

    def publication_blockers(self) -> list[str]:
        """Blockers durs en mode publication (règle architecturale centrale)."""
        b: list[str] = []
        if self.background_mode == "source_background":
            b.append("source_background_forbidden")
        if not self.clean_background_path:
            b.append("missing_clean_background")
        if self.clean_background_verified is not True:
            b.append("clean_background_not_verified")
        if self.source_text_leak_risk in {"high", "critical"}:
            b.append("source_text_leak")
        return b

    @classmethod
    def from_resolved(cls, bg: dict, assets: dict | None = None) -> "BackgroundContract":
        assets = assets or {}
        mode = bg.get("mode") or "blank_degraded"
        clean = bg.get("path") if mode == "clean_background" else (
            assets.get("background_path") or assets.get("background_clean_path"))
        risk = bg.get("source_text_leak_risk") or "high"
        verified = bool(bg.get("clean_background_verified"))
        return cls(
            clean_background_path=clean,
            source_image_path=assets.get("source_image_path") or (bg.get("path") if mode == "source_background" else None),
            background_mode=mode,
            # Never infer text removal from the mere presence of a cleanbg file.
            # The renderer may still need patches when clean_background_verified is false.
            text_removed=bool(bg.get("text_removed")) and verified,
            clean_background_verified=verified,
            source_text_leak_risk=risk,
            publication_allowed=(mode == "clean_background" and clean is not None and verified and risk in {"low", "medium"}),
            findings=list(bg.get("findings") or []),
        )
