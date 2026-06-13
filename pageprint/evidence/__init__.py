"""Evidence collection and resolution for PAGEPRINT."""

from .claim_model import make_claim
from .collector import collect_claims
from .resolver import resolve_all, resolve_unit_evidence

__all__ = [
    "make_claim",
    "collect_claims",
    "resolve_all",
    "resolve_unit_evidence",
]
