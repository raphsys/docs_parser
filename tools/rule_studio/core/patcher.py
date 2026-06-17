"""Génération et application sécurisée de patches (§9, §12).

Principe : jamais d'application silencieuse. Le flux est toujours
    proposer -> diff -> (validation humaine) -> appliquer -> tests -> rapport.

Pour la V1, l'injection automatique de code dans le pipeline est volontairement
limitée. Le patcher sait :
    1. matérialiser une règle gérée dans `managed_rules.yaml` ;
    2. produire un patch « commentaire de règle » à insérer près de la cible ;
    3. générer un diff unifié pour relecture ;
    4. appliquer un diff après confirmation, avec backup Git.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path

from . import git_guard
from .models import RuleRecord


@dataclass
class PatchProposal:
    rule_id: str
    target_file: str
    description: str
    original: str
    proposed: str
    diff: str
    safe: bool = True
    reason: str = "OK"
    applied: bool = False
    backup_branch: str = ""
    warnings: list[str] = field(default_factory=list)


def _unified_diff(original: str, proposed: str, path: str) -> str:
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            proposed.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def _rule_comment_block(rec: RuleRecord) -> str:
    """Bloc commentaire structuré documentant la règle dans le code cible."""
    lines = [
        f"# RULE-ID: {rec.id}",
        f"# RULE-UNIT: {rec.pipeline_unit}",
        f"# RULE-TYPE: {rec.rule_type}",
        f"# RULE-OBJECTIVE: {rec.objective or rec.title}",
        f"# RULE-STATUS: {rec.status}",
        f"# RULE-MANAGED: {'true' if rec.managed else 'false'}",
    ]
    for t in rec.tests_linked:
        lines.append(f"# RULE-TEST: {t}")
    lines.append("# END-RULE")
    return "\n".join(lines) + "\n"


def propose_comment_patch(rec: RuleRecord, root: str | Path) -> PatchProposal:
    """Propose d'insérer un bloc commentaire de règle en tête du fichier cible.

    Approche non intrusive : on documente la règle dans le code sans modifier la
    logique. C'est le mode sûr par défaut de la V1.
    """
    root = Path(root)
    target = rec.target.get("file") or rec.file_path
    if not target:
        return PatchProposal(
            rule_id=rec.id,
            target_file="",
            description="Aucun fichier cible.",
            original="",
            proposed="",
            diff="",
            safe=False,
            reason="Règle sans fichier cible : patch impossible.",
        )

    fpath = root / target
    original = fpath.read_text(encoding="utf-8") if fpath.exists() else ""
    block = _rule_comment_block(rec)

    if f"RULE-ID: {rec.id}" in original:
        return PatchProposal(
            rule_id=rec.id,
            target_file=target,
            description="Bloc déjà présent.",
            original=original,
            proposed=original,
            diff="",
            safe=True,
            reason="Le bloc de règle existe déjà : aucune modification.",
        )

    # Insertion après l'éventuel docstring/imports : ici, en tête de fichier
    # après la première ligne shebang/encoding si présente.
    insert_at = 0
    olines = original.splitlines(keepends=True)
    if olines and olines[0].startswith("#!"):
        insert_at = 1
    proposed = "".join(olines[:insert_at]) + block + "".join(olines[insert_at:])

    state = git_guard.inspect(root)
    safe, reason = git_guard.can_safely_patch(state, target)
    return PatchProposal(
        rule_id=rec.id,
        target_file=target,
        description=f"Insérer le bloc commentaire de la règle {rec.id}.",
        original=original,
        proposed=proposed,
        diff=_unified_diff(original, proposed, target),
        safe=safe,
        reason=reason,
    )


def apply_patch(
    proposal: PatchProposal,
    root: str | Path,
    confirm: bool,
    make_backup: bool = True,
) -> PatchProposal:
    """Applique un patch APRÈS confirmation explicite (§12 : jamais silencieux)."""
    if not confirm:
        proposal.warnings.append("Application refusée : confirmation manquante.")
        return proposal
    if not proposal.safe:
        proposal.warnings.append(f"Application bloquée : {proposal.reason}")
        return proposal
    if not proposal.diff:
        proposal.warnings.append("Aucun changement à appliquer.")
        return proposal

    root = Path(root)
    if make_backup:
        try:
            proposal.backup_branch = git_guard.create_backup_branch(root)
        except RuntimeError as exc:
            proposal.warnings.append(f"Backup impossible : {exc}")

    fpath = root / proposal.target_file
    fpath.write_text(proposal.proposed, encoding="utf-8")
    proposal.applied = True
    return proposal


def rollback(proposal: PatchProposal, root: str | Path) -> bool:
    """Restaure le contenu original du fichier patché."""
    if not proposal.applied:
        return False
    fpath = Path(root) / proposal.target_file
    fpath.write_text(proposal.original, encoding="utf-8")
    proposal.applied = False
    return True
