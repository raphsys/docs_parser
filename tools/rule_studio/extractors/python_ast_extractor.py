"""Extracteur de règles depuis du code Python via le module `ast`.

Principe directeur (cf. cahier des charges §6.1) : tout `if` n'est PAS une règle.
On attribue à chaque candidat un score de confiance fondé sur des signaux forts
(nom de fichier/fonction, vocabulaire métier, seuils numériques). En dessous d'un
plancher, le candidat est rejeté pour éviter le bruit.

Constructions extraites :
    if / elif, match/case, assert, raise (ValueError/AssertionError),
    Enum, dataclass, constantes MAJUSCULES, dictionnaires/listes de règles,
    fonctions « métier » (classify/resolve/validate/compile/detect/score/is_*).
"""

from __future__ import annotations

import ast
import re

from ..core.classifier import classify
from ..core.models import RuleRecord, make_rule_id

# --- Vocabulaire métier : signal fort de « vraie règle pipeline » -----------
_DOMAIN_TERMS = re.compile(
    r"\b(role|page_role|layout_type|translatable|render_policy|bbox|protected|"
    r"anchor|overflow|cover|family|policy|constraint|fallback|validate|"
    r"reconstruct|paragraph_flow|anchored_text|visual_density|text_density|"
    r"image_dominant|font|typograph|formula|caption|threshold|score|quality)\b",
    re.IGNORECASE,
)

_FN_VERBS = re.compile(
    r"(classify|resolve|validate|compile|detect|score|fallback|"
    r"is_[a-z]|policy|rule|constraint|segment|protect|reconstruct)",
    re.IGNORECASE,
)

_FILE_HINT = re.compile(
    r"(rule|policy|constraint|validator|classifier|resolver|quality|"
    r"compiler|contract|detector|protection)",
    re.IGNORECASE,
)

_NUMERIC_CMP = re.compile(r"[<>]=?|==")

# Plancher de confiance en dessous duquel on n'émet pas de règle.
_MIN_CONFIDENCE = 0.30


def _has_domain(text: str) -> bool:
    return bool(_DOMAIN_TERMS.search(text))


def _guess_rule_type(symbol: str, code: str, source: str) -> str:
    blob = f"{symbol} {source} {code}".lower()
    table = [
        ("contract", r"contract"),
        ("validator", r"validat|assert|raise"),
        ("policy", r"policy"),
        ("constraint", r"constraint"),
        ("classification", r"classif|page_role|family|cover"),
        ("segmentation", r"segment|sentence_boundary|coalesc"),
        ("translation", r"translat|terminology"),
        ("protected_content", r"protect|non.?translat|formula"),
        ("reconstruction", r"reconstruct|patch|plan_compiler"),
        ("typography", r"font|typograph|text_measure"),
        ("layout", r"layout|placement|bbox|anchor|overflow"),
        ("quality_score", r"quality|score|coverage"),
        ("threshold", r"threshold|_min|_max|>=|<="),
        ("fallback", r"fallback|default"),
        ("legacy_compatibility", r"legacy|_v2|_v3|compat"),
    ]
    for rtype, pat in table:
        if re.search(pat, blob):
            return rtype
    return "heuristic"


class _RuleVisitor(ast.NodeVisitor):
    def __init__(self, source: str, file_path: str) -> None:
        self.source = source
        self.lines = source.splitlines()
        self.file_path = file_path
        self.rules: list[RuleRecord] = []
        self._fn_stack: list[str] = []
        self._cls_stack: list[str] = []

    # -- contexte ------------------------------------------------------------
    def _current_symbol(self) -> str | None:
        parts = self._cls_stack + self._fn_stack
        return ".".join(parts) if parts else None

    def _segment(self, node: ast.AST) -> str:
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if not start:
            return ""
        end = end or start
        return "\n".join(self.lines[start - 1 : end])

    # -- émission ------------------------------------------------------------
    def _emit(self, node: ast.AST, kind: str, base_signal: float) -> None:
        code = self._segment(node)
        if not code:
            return
        symbol = self._current_symbol()

        # Signal : nom de fichier + symbole + vocabulaire métier + comparaison.
        signal = base_signal
        if _FILE_HINT.search(self.file_path):
            signal += 0.25
        if symbol and _FN_VERBS.search(symbol):
            signal += 0.20
        if _has_domain(code) or (symbol and _has_domain(symbol)):
            signal += 0.30
        if _NUMERIC_CMP.search(code):
            signal += 0.10

        unit, secondary, unit_conf = classify(self.file_path, symbol, code)
        confidence = min(1.0, round(signal * 0.5 + unit_conf * 0.5, 3))
        if confidence < _MIN_CONFIDENCE:
            return

        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        rtype = _guess_rule_type(symbol or "", code, self.source[:0])

        first = code.strip().splitlines()[0][:120] if code.strip() else kind
        title = f"[{kind}] {first}"

        rid = make_rule_id("RS", f"{self.file_path}:{start}:{first}")
        rec = RuleRecord(
            id=rid,
            title=title,
            description=f"Règle {kind} détectée dans {symbol or self.file_path}.",
            pipeline_unit=unit,
            secondary_units=secondary,
            rule_type=rtype,
            source_type="python_ast",
            file_path=self.file_path,
            line_start=start,
            line_end=end,
            symbol_name=symbol,
            raw_code=code[:2000],
            normalized_expression=first,
            status="candidate",
            usage_status="unknown",
            validation_status="untested",
            confidence=confidence,
            risk_level="medium" if confidence >= 0.6 else "low",
            managed=False,
            editable=True,
            deletable=True,
            simulable=False,
        )
        rec.last_seen_hash = rec.compute_hash()
        rec.touch()
        self.rules.append(rec)

    # -- visiteurs -----------------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._fn_stack.append(node.name)
        # Une fonction « métier » entière peut constituer une règle.
        if _FN_VERBS.search(node.name) and _has_domain(self._segment(node)):
            self._emit(node, "function", base_signal=0.35)
        self.generic_visit(node)
        self._fn_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._cls_stack.append(node.name)
        bases = {getattr(b, "id", getattr(b, "attr", "")) for b in node.bases}
        is_model = bool(bases & {"Enum", "IntEnum", "BaseModel", "TypedDict"})
        is_model = is_model or any(
            isinstance(d, ast.Name) and d.id == "dataclass"
            for d in node.decorator_list
        )
        if is_model or _has_domain(node.name):
            self._emit(node, "schema", base_signal=0.30)
        self.generic_visit(node)
        self._cls_stack.pop()

    def visit_If(self, node: ast.If) -> None:
        test_code = self._segment(node.test) if hasattr(node, "test") else ""
        if _has_domain(test_code) and _NUMERIC_CMP.search(test_code + " x"):
            self._emit(node, "branch", base_signal=0.20)
        elif _has_domain(test_code):
            self._emit(node, "branch", base_signal=0.12)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        subj = self._segment(node.subject)
        if _has_domain(subj) or _has_domain(self._segment(node)):
            self._emit(node, "match", base_signal=0.25)
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        if _has_domain(self._segment(node)):
            self._emit(node, "assert", base_signal=0.30)
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        exc = node.exc
        name = ""
        if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
            name = exc.func.id
        if name in {"ValueError", "AssertionError"} and _has_domain(
            self._segment(node)
        ):
            self._emit(node, "raise", base_signal=0.25)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # Constantes MAJUSCULES + dicts/listes de règles.
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id.isupper() and len(tgt.id) > 2:
                seg = self._segment(node)
                if _has_domain(seg) or isinstance(
                    node.value, (ast.Dict, ast.List, ast.Set, ast.Tuple)
                ):
                    self._emit(node, "constant", base_signal=0.20)
                break
        self.generic_visit(node)


def extract_python_rules(source: str, file_path: str) -> list[RuleRecord]:
    """Renvoie les `RuleRecord` détectés dans un fichier Python."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    visitor = _RuleVisitor(source, file_path)
    visitor.visit(tree)
    # Déduplication par (fichier, ligne, titre)
    seen: set[tuple] = set()
    out: list[RuleRecord] = []
    for r in visitor.rules:
        key = (r.file_path, r.line_start, r.title)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out
