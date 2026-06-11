"""PageUnderstanding — unité 4 du pipeline cible.

Compréhension de page avant PAGEPRINT :
    - LayoutV2Builder (hub de page understanding : page_role, page_family,
      document_type, layout_type, style_profile, page_case, page_case_v2,
      relations d'éléments) ;
    - special_region_detector (optionnel — import fail-safe) ;
    - page_extraction_postprocessors + rebuild LayoutV2 si la structure change.

Chaque module est optionnel et fail-safe : si un import échoue, le pipeline
continue et la trace l'indique.
"""

from __future__ import annotations


class PageUnderstanding:
    def __init__(self, *, enable_layout_v2: bool = True,
                 enable_postprocessors: bool = True,
                 enable_special_regions: bool = True):
        self.enable_layout_v2 = enable_layout_v2
        self.enable_postprocessors = enable_postprocessors
        self.enable_special_regions = enable_special_regions
        self._layout_builder = None

    def _layout_v2(self):
        if self._layout_builder is None:
            from structure_extractor import LayoutV2Builder

            self._layout_builder = LayoutV2Builder()
        return self._layout_builder

    def apply(self, page_structure: dict, *, img=None, pdf_page=None,
              sx: float = 1.0, sy: float = 1.0) -> tuple[dict, dict]:
        trace: dict = {}

        if self.enable_layout_v2:
            try:
                page_structure = self._layout_v2().build(page_structure)
                trace["layout_v2"] = {"applied": True}
            except Exception as exc:
                trace["layout_v2"] = {"applied": False, "error": str(exc)}

        if self.enable_special_regions:
            page_structure, sr_trace = self._apply_special_regions(
                page_structure, img=img, pdf_page=pdf_page, sx=sx, sy=sy
            )
            trace["special_region_detector"] = sr_trace

        if self.enable_postprocessors:
            try:
                from page_extraction_postprocessors import (
                    apply_page_extraction_postprocessors,
                )

                page_structure, postprocess_info = apply_page_extraction_postprocessors(
                    page_structure
                )
                trace["postprocessors"] = {"applied": True, "info": _summarize(postprocess_info)}
                if isinstance(postprocess_info, dict) and (
                    postprocess_info.get("changed") or postprocess_info.get("applied")
                ) and self.enable_layout_v2:
                    try:
                        page_structure = self._layout_v2().build(page_structure)
                        trace["layout_v2_rebuild"] = {"applied": True}
                    except Exception as exc:
                        trace["layout_v2_rebuild"] = {"applied": False, "error": str(exc)}
            except Exception as exc:
                trace["postprocessors"] = {"applied": False, "error": str(exc)}

        page_structure.setdefault("pipeline_trace", {})
        page_structure["pipeline_trace"]["page_understanding"] = trace
        return page_structure, trace

    @staticmethod
    def _apply_special_regions(page_structure: dict, *, img=None, pdf_page=None,
                               sx: float = 1.0, sy: float = 1.0) -> tuple[dict, dict]:
        try:
            from special_region_detector import detect_special_regions
        except Exception as exc:
            return page_structure, {"available": False, "error": str(exc)}

        try:
            enriched, report = detect_special_regions(
                page_structure, page_image=img, pdf_page=pdf_page, sx=sx, sy=sy
            )
        except TypeError:
            # Signature alternative (structure seule).
            try:
                enriched, report = detect_special_regions(page_structure)
            except Exception as exc:
                return page_structure, {"available": True, "applied": False, "error": str(exc)}
        except Exception as exc:
            return page_structure, {"available": True, "applied": False, "error": str(exc)}

        if isinstance(enriched, dict):
            special_regions = enriched.get("special_regions") or []
            page_structure["special_regions"] = special_regions
            page_structure.setdefault("layout", {})
            page_structure["layout"]["special_regions"] = special_regions
            return page_structure, {
                "available": True,
                "applied": True,
                "special_region_count": len(special_regions),
                "report": _summarize(report),
            }
        return page_structure, {"available": True, "applied": False}


def _summarize(info):
    """Trace compacte : ne garder que les scalaires de premier niveau."""
    if not isinstance(info, dict):
        return info
    return {
        k: v for k, v in info.items()
        if isinstance(v, (str, int, float, bool)) or v is None
    }
