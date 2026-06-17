"""Politique de réessai de l'AutoCorrectionLoop."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetryPolicy:
    max_iter: int = 3
    no_repeat_same_failure: bool = True
    no_quality_degradation_without_gain: bool = True
    no_protected_overlap_allowed: bool = True
    min_gain: float = 0.01
