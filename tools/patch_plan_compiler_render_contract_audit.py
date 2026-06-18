#!/usr/bin/env python3
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
old = s

if "render_contract_audit" in s and "render_contract_propagation_conflict" in s:
    print(f"déjà patché: {p}")
    raise SystemExit(0)

target = '        try:\n            ownership_audit = audit_source_ownership(plan.to_dict(), normalized)\n            plan.render_policy["source_ownership_audit"] = {\n                "status": ownership_audit.get("status"),\n                "conflict_count": ownership_audit.get("conflict_count"),\n                "hard_blockers": ownership_audit.get("hard_blockers") or [],\n            }\n            if ownership_audit.get("status") != "ok":\n                findings.append({"type": "source_ownership_conflict", "severity": "ko", "detail": ownership_audit})\n        except Exception as exc:  # pragma: no cover\n            findings.append({"type": "source_ownership_audit_failed", "message": str(exc), "severity": "review"})\n'
insert = '        try:\n            ownership_audit = audit_source_ownership(plan.to_dict(), normalized)\n            plan.render_policy["source_ownership_audit"] = {\n                "status": ownership_audit.get("status"),\n                "conflict_count": ownership_audit.get("conflict_count"),\n                "hard_blockers": ownership_audit.get("hard_blockers") or [],\n            }\n            if ownership_audit.get("status") != "ok":\n                findings.append({"type": "source_ownership_conflict", "severity": "ko", "detail": ownership_audit})\n        except Exception as exc:  # pragma: no cover\n            findings.append({"type": "source_ownership_audit_failed", "message": str(exc), "severity": "review"})\n\n        # Ownership/Lifecycle v2: verify that preserved_visual ownership really\n        # propagates to the render contract: protected region + preserved layer +\n        # PreservationOp, with no leak into translation units, translated_text,\n        # TextOp or destructive PatchOp.  This is a hard diagnostic gate; it does\n        # not repair layout, it prevents silent contract regression.\n        try:\n            from render_contract_audit import audit_render_contract, compact_render_contract_audit\n            render_contract_audit = audit_render_contract(plan.to_dict(), normalized)\n            plan.render_policy["render_contract_audit"] = compact_render_contract_audit(render_contract_audit)\n            if render_contract_audit.get("status") != "ok":\n                findings.append({\n                    "type": "render_contract_propagation_conflict",\n                    "severity": "ko",\n                    "detail": compact_render_contract_audit(render_contract_audit, max_rows=10),\n                })\n        except Exception as exc:  # pragma: no cover\n            findings.append({"type": "render_contract_audit_failed", "message": str(exc), "severity": "review"})\n'
if target not in s:
    raise SystemExit("bloc source_ownership_audit introuvable dans pagereconstruct/plan_compiler.py")
s = s.replace(target, insert, 1)
p.write_text(s, encoding="utf-8")
print(f"corrigé: {p}")
