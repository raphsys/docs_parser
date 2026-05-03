"""
P6 — Agent d'audit du fond (background master).

Analyse les résidus visuels potentiels dans le bg_master et détermine :
- Si des artéfacts textuels sont encore visibles (texte source non effacé)
- Les zones à re-inpainter
- Si l'inpainting existant est satisfaisant

Note : cet agent travaille sur des métadonnées textuelles, pas sur des pixels.
L'analyse réelle des pixels reste dans background_inpainter.py.
"""

from __future__ import annotations

import json
from typing import Any

from .base import ModelRuntime, PipelineAgent, _extract_json


_SYSTEM_PROMPT = """\
You are a document background quality expert. Return JSON only.
Output schema: {"quality":0.0,"artifacts":[...],"reprocess":[...],"ok":true|false}
- quality: float 0.0–1.0 (1.0 = clean background, no visible artifacts)
- artifacts: [{"region":[x0,y0,x1,y1],"type":"text_residue|shadow|smudge|bleed","severity":"low|medium|high"}]
- reprocess: [[x0,y0,x1,y1],...] — regions that need re-inpainting
- ok: true if quality >= 0.85 and no high-severity artifacts

Rules:
- text_residue: source text still partially visible
- shadow: drop shadow from removed text
- smudge: ink bleed or smear
- bleed: content from adjacent page visible
- Mark reprocess only for medium or high severity artifacts
- ok=false if any critical artifact exists

If unsure: {"quality":0.9,"artifacts":[],"reprocess":[],"ok":true}"""


_FEW_SHOT: list[tuple[str, str]] = [
    (
        '{"page_id":"p1","blocks_removed":5,"inpaint_regions":[[10,20,200,40],[15,60,180,80]],"coverage_ratio":0.92,"avg_confidence":0.88}',
        '{"quality":0.90,"artifacts":[],"reprocess":[],"ok":true}',
    ),
    (
        '{"page_id":"p2","blocks_removed":12,"inpaint_regions":[[10,20,400,350]],"coverage_ratio":0.45,"avg_confidence":0.55}',
        '{"quality":0.55,"artifacts":[{"region":[10,20,400,350],"type":"text_residue","severity":"high"}],"reprocess":[[10,20,400,350]],"ok":false}',
    ),
]

_MAX_FEW_SHOT = 2


class P6BackgroundAgent(PipelineAgent):
    """
    Agent P6 : audit qualité du background master.

    Entrée (``input_data``) :
    ```json
    {
      "page_id": "p1",
      "blocks_removed": 8,
      "inpaint_regions": [[x0, y0, x1, y1], ...],
      "coverage_ratio": 0.88,
      "avg_confidence": 0.82
    }
    ```

    Sortie :
    ```json
    {
      "quality": 0.88,
      "artifacts": [],
      "reprocess": [],
      "ok": true
    }
    ```
    """

    stage = "p6_background"
    prompt_version = "v1"
    default_max_new_tokens = 150

    _VALID_ARTIFACT_TYPES = {"text_residue", "shadow", "smudge", "bleed"}
    _VALID_SEVERITIES = {"low", "medium", "high"}

    def build_messages(self, input_data: dict) -> list[dict]:
        block_json = json.dumps(input_data, ensure_ascii=False, separators=(",", ":"))
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for user_ex, asst_ex in _FEW_SHOT[:_MAX_FEW_SHOT]:
            messages.append({"role": "user", "content": user_ex})
            messages.append({"role": "assistant", "content": asst_ex})
        messages.append({"role": "user", "content": block_json})
        return messages

    def parse_response(self, raw: str, input_data: dict) -> dict:
        default: dict = {"quality": 0.9, "artifacts": [], "reprocess": [], "ok": True}
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            return default

        try:
            quality = float(parsed.get("quality") or 0.9)
            quality = max(0.0, min(1.0, quality))
        except (TypeError, ValueError):
            quality = 0.9

        raw_artifacts = parsed.get("artifacts") or []
        valid_artifacts: list[dict] = []
        for art in raw_artifacts:
            if not isinstance(art, dict):
                continue
            atype = str(art.get("type") or "").lower()
            severity = str(art.get("severity") or "low").lower()
            region = art.get("region")
            if atype not in self._VALID_ARTIFACT_TYPES:
                continue
            if severity not in self._VALID_SEVERITIES:
                continue
            entry: dict = {"type": atype, "severity": severity}
            if isinstance(region, list) and len(region) == 4:
                try:
                    entry["region"] = [float(x) for x in region]
                except (TypeError, ValueError):
                    pass
            valid_artifacts.append(entry)

        raw_reprocess = parsed.get("reprocess") or []
        valid_reprocess: list[list[float]] = []
        for reg in raw_reprocess:
            if isinstance(reg, list) and len(reg) == 4:
                try:
                    valid_reprocess.append([float(x) for x in reg])
                except (TypeError, ValueError):
                    continue

        ok_raw = parsed.get("ok")
        if isinstance(ok_raw, bool):
            ok = ok_raw
        else:
            ok = quality >= 0.85 and not any(
                a["severity"] == "high" for a in valid_artifacts
            )

        return {
            "quality": quality,
            "artifacts": valid_artifacts,
            "reprocess": valid_reprocess,
            "ok": ok,
        }

    @classmethod
    def build_input_from_page(cls, page_data: dict) -> dict:
        """Construit l'input de l'agent depuis les données d'une page."""
        regions = [
            b.get("bbox")
            for b in (page_data.get("blocks") or [])
            if b.get("bbox") and b.get("inpainted")
        ]
        confidences = [
            float(b.get("inpaint_confidence") or 0.0)
            for b in (page_data.get("blocks") or [])
            if b.get("inpainted")
        ]
        avg_conf = sum(confidences) / max(1, len(confidences)) if confidences else 0.0
        total_blocks = len(page_data.get("blocks") or [])
        inpainted = len(regions)
        return {
            "page_id": str(page_data.get("page_id") or page_data.get("id") or ""),
            "blocks_removed": inpainted,
            "inpaint_regions": regions,
            "coverage_ratio": round(inpainted / max(1, total_blocks), 3),
            "avg_confidence": round(avg_conf, 3),
        }
