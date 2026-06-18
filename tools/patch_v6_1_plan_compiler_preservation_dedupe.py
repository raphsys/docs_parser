#!/usr/bin/env python3
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
old = s

old_block = '''    underlays, overlays = [], []
    preserve_index = 1
    for zidx, zone in enumerate(special_zones or [], start=1):
        pu = PreservedUnit(
            id=f"pres_special_{zidx:04d}",
            source="special_zone",
            reason=zone.get("kind") or "protected_visual",
            bbox=zone.get("bbox"),
            text=None,
            preservation_mode="preserve_as_visual_overlay",
            source_unit_ids=list(zone.get("source_unit_ids") or []),
            z_policy="preserve_original",
        )
        underlays.append(pu)
        for sid in zone.get("source_unit_ids") or []:
            excluded.add(sid)
            rendered_ids.add(sid)
        findings.append({
            "type": "special_zone_preserved_as_underlay",
            "zone_kind": zone.get("kind"),
            "bbox": zone.get("bbox"),
            "source_unit_ids": list(zone.get("source_unit_ids") or []),
            "severity": "info",
        })
    preserve_index = len(underlays) + 1
'''
new_block = '''    underlays, overlays = [], []

    # Existing preservation_plan entries already represent some formula/code/
    # protected visual objects. Special detector zones strengthen protection,
    # but must not create a second preserved layer for the same source object.
    existing_preserve_sids = set()
    existing_preserve_boxes = []
    for _p in preservation_plan or []:
        existing_preserve_sids.update(_p.get("source_unit_ids") or [])
        _b = _p.get("bbox")
        if isinstance(_b, (list, tuple)) and len(_b) == 4:
            existing_preserve_boxes.append([float(x) for x in _b])

    def _zone_already_preserved(zone: dict) -> bool:
        z_sids = set(zone.get("source_unit_ids") or [])
        if z_sids and (z_sids & existing_preserve_sids):
            return True
        z_box = zone.get("bbox")
        if not (isinstance(z_box, (list, tuple)) and len(z_box) == 4):
            return False
        z_box = [float(x) for x in z_box]
        for _b in existing_preserve_boxes:
            if _contained_ratio(z_box, _b) >= 0.80 and _contained_ratio(_b, z_box) >= 0.80:
                return True
        return False

    preserve_index = 1
    for zidx, zone in enumerate(special_zones or [], start=1):
        if _zone_already_preserved(zone):
            for sid in zone.get("source_unit_ids") or []:
                excluded.add(sid)
                rendered_ids.add(sid)
            findings.append({
                "type": "special_zone_preservation_deduped",
                "zone_kind": zone.get("kind"),
                "bbox": zone.get("bbox"),
                "source_unit_ids": list(zone.get("source_unit_ids") or []),
                "severity": "info",
            })
            continue
        pu = PreservedUnit(
            id=f"pres_special_{zidx:04d}",
            source="special_zone",
            reason=zone.get("kind") or "protected_visual",
            bbox=zone.get("bbox"),
            text=None,
            preservation_mode="preserve_as_visual_overlay",
            source_unit_ids=list(zone.get("source_unit_ids") or []),
            z_policy="preserve_original",
        )
        underlays.append(pu)
        for sid in zone.get("source_unit_ids") or []:
            excluded.add(sid)
            rendered_ids.add(sid)
        findings.append({
            "type": "special_zone_preserved_as_underlay",
            "zone_kind": zone.get("kind"),
            "bbox": zone.get("bbox"),
            "source_unit_ids": list(zone.get("source_unit_ids") or []),
            "severity": "info",
        })
    preserve_index = len(underlays) + 1
'''
if old_block in s:
    s = s.replace(old_block, new_block, 1)
elif "def _zone_already_preserved(" not in s:
    raise SystemExit("bloc special_zones à patcher introuvable")

if s != old:
    p.write_text(s, encoding="utf-8")
    print(f"corrigé: {p}")
else:
    print(f"aucune modification nécessaire: {p}")
