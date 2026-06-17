"""AutoCorrectionLoop — la validation CORRIGE, pas seulement constate.

Boucle générique et testable : compile → audit → si échec corrigeable, applique
une correction (knob) → recompile → garde le MEILLEUR score (net-improvement),
max_iter, jamais d'overlap protégé. Les fonctions compile/audit sont injectées
pour rester découplées du pipeline lourd.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .retry_policy import RetryPolicy
from .correction_plan import plan_corrections


@dataclass
class LoopResult:
    best_plan: dict
    best_report: object
    iterations: int = 0
    applied: list = field(default_factory=list)     # [action]
    history: list = field(default_factory=list)      # [(iter, score, status)]

    def to_dict(self):
        return {"iterations": self.iterations, "applied": self.applied,
                "history": self.history,
                "final_score": getattr(self.best_report, "publication_ready_score", None),
                "final_status": getattr(self.best_report, "status", None)}


def run_correction_loop(compile_fn, audit_fn, *, knobs0: dict | None = None,
                        policy: RetryPolicy | None = None):
    """compile_fn(knobs)->plan ; audit_fn(plan)->PagePublicationReadyReport.
    knobs: dict de réglages (ex: {'shrink': 0.0, 'multiblock': True})."""
    policy = policy or RetryPolicy()
    knobs = dict(knobs0 or {})
    plan = compile_fn(knobs)
    report = audit_fn(plan)
    best = LoopResult(best_plan=plan, best_report=report)
    best.history.append((0, getattr(report, "publication_ready_score", 0.0), getattr(report, "status", "")))
    seen_failures = set()

    for i in range(1, policy.max_iter + 1):
        if getattr(report, "publication_ready", False):
            break
        actions = plan_corrections(report)
        # ne pas répéter le même échec
        actions = [a for a in actions if not (policy.no_repeat_same_failure and a.reason in seen_failures)]
        if not actions:
            break
        for a in actions:
            seen_failures.add(a.reason)
            _apply_knob(knobs, a)
        cand_plan = compile_fn(knobs)
        cand_report = audit_fn(cand_plan)
        # garde net-improvement, jamais accepter overlap protégé
        gain = getattr(cand_report, "publication_ready_score", 0.0) - getattr(best.best_report, "publication_ready_score", 0.0)
        protected_ok = "block_protected_overlap" not in cand_report.hard_blockers and \
                       "patch_protected_overlap" not in cand_report.hard_blockers
        best.iterations = i
        best.history.append((i, getattr(cand_report, "publication_ready_score", 0.0), getattr(cand_report, "status", "")))
        if protected_ok and gain >= policy.min_gain:
            best.best_plan, best.best_report, report = cand_plan, cand_report, cand_report
            best.applied.extend(a.action for a in actions)
        else:
            report = cand_report  # continuer à diagnostiquer mais ne pas régresser le best
    return best


def _apply_knob(knobs: dict, action) -> None:
    a = action.action
    if a == "shrink_block":
        knobs["shrink"] = min(0.14, knobs.get("shrink", 0.0) + 0.07)
    elif a == "reflow_block" or a == "move_block":
        knobs["multiblock"] = True
    elif a == "adjust_line_height":
        knobs["compact_line_height"] = True
    elif a == "force_code_preserve":
        knobs["force_code_preserve"] = True
    elif a == "regenerate_background_zone":
        knobs["regenerate_background"] = True
    elif a == "mark_review":
        knobs["mark_review"] = True
