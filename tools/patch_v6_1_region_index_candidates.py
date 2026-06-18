#!/usr/bin/env python3
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
old = s

s = s.replace('"formula_candidate_region": "formula_region",', '"formula_candidate_region": "formula_candidate_region",')
s = s.replace('"code_candidate_region": "code_region",', '"code_candidate_region": "code_candidate_region",')
s = s.replace('"visual_candidate_region": "protected_visual_region",', '"visual_candidate_region": "visual_candidate_region",')

if "def _confirmed_region_type_from_candidate(" not in s:
    marker = "\ndef build_regions("
    helper = '''
def _confirmed_region_type_from_candidate(region_type: str) -> str | None:
    if region_type == "formula_candidate_region":
        return "formula_region"
    if region_type == "code_candidate_region":
        return "code_region"
    if region_type == "visual_candidate_region":
        return "protected_visual_region"
    return None


def _make_confirmed_from_candidate(region: dict, confirmed_type: str, prefix: str, counter: int) -> dict:
    confirmed = dict(region)
    confirmed["region_id"] = f"{prefix}_region_{confirmed_type}_{counter:03d}"
    confirmed["region_type"] = confirmed_type
    confirmed["role"] = confirmed_type
    confirmed["source"] = region.get("source") or "candidate_promoted"
    confirmed["policy"] = _region_policy(confirmed_type)
    confirmed["constraints"] = _region_constraints(confirmed_type)
    obj = "formula" if confirmed_type == "formula_region" else "code" if confirmed_type == "code_region" else "protected_visual"
    confirmed.update({
        "object_type": region.get("object_type") or obj,
        "object_class": region.get("object_class") or obj,
        "claim_type": f"{obj}_confirmed_from_candidate",
        "policy_pending": False,
        "observation_only": False,
        "protected_visual": True,
        "preserve_original_pixels": True,
        "reason": "candidate_promoted_to_hard_special_region",
        "detection_source": region.get("detection_source") or region.get("source") or "pageprint_candidate",
        "promoted_from_region_id": region.get("region_id"),
    })
    return confirmed

'''
    if marker in s:
        s = s.replace(marker, "\n" + helper + marker, 1)
    else:
        raise SystemExit("build_regions marker introuvable")

needle = '''        regions.append(region)

    return regions
'''
replacement = '''        regions.append(region)
        confirmed_type = _confirmed_region_type_from_candidate(region_type)
        if confirmed_type:
            type_counters[confirmed_type] = type_counters.get(confirmed_type, 0) + 1
            regions.append(_make_confirmed_from_candidate(region, confirmed_type, prefix, type_counters[confirmed_type]))

    return regions
'''
if needle in s and "_make_confirmed_from_candidate(region, confirmed_type" not in s:
    s = s.replace(needle, replacement, 1)

if s != old:
    p.write_text(s, encoding="utf-8")
    print(f"corrigé: {p}")
else:
    print(f"aucune modification nécessaire: {p}")
