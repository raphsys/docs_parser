"""Rule Studio — console Streamlit de gouvernance des règles du pipeline.

Lancement :
    streamlit run tools/rule_studio/app.py

L'application scanne le projet, affiche les règles dans un tableau éditable,
permet filtres, création, simulation, diff et application sécurisée de patches,
puis exporte l'inventaire et un rapport d'audit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Permet `python -m streamlit run` depuis n'importe où : on ajoute la racine du
# dépôt au sys.path pour résoudre le package `tools.rule_studio`.
_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.rule_studio.core import patcher  # noqa: E402
from tools.rule_studio.core.models import (  # noqa: E402
    BOOL_DECISION_FIELDS,
    KEEP_DECISIONS,
    LIFECYCLE_STATUSES,
    PIPELINE_UNITS,
    RISK_LEVELS,
    RULE_TYPES,
    STATUSES,
    USAGE_STATUSES,
    VALIDATION_STATUSES,
    RuleRecord,
    make_rule_id,
)
from tools.rule_studio.core.simulator import simulate, validate_dsl  # noqa: E402
from tools.rule_studio.core.studio import (  # noqa: E402
    DEFAULT_DB,
    DEFAULT_MANAGED,
    Studio,
)
from tools.rule_studio.core.test_runner import run_tests  # noqa: E402

st.set_page_config(page_title="Rule Studio", layout="wide")

# Colonnes affichées dans le tableau (ordre). Inclut le langage naturel, les
# cases à cocher de décision et le cycle de vie (directive §8.1).
DISPLAY_COLS = [
    "id",
    "pipeline_unit",
    "rule_type",
    "natural_language_title",
    "natural_language_rule",
    "objective",
    "understanding",
    "human_decision",
    "recommended_decision",
    "proposed_modification",
    *BOOL_DECISION_FIELDS,            # keep, active, selected, modify, delete, ...
    "lifecycle_status",
    "validation_status",
    "usage_status",
    "risk_level",
    "file_path",
    "line_start",
    "confidence",
    "source_type",
    "ai_interpreted",
    "managed",
]

# Champs texte éditables (langage naturel + décision libre).
EDITABLE_TEXT_COLS = {
    "natural_language_title",
    "natural_language_rule",
    "objective",
    "understanding",
    "human_decision",
    "proposed_modification",
}
# Toutes les colonnes modifiables = texte naturel + cases booléennes.
EDITABLE_COLS = EDITABLE_TEXT_COLS | set(BOOL_DECISION_FIELDS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_resource
def get_studio(project_root: str) -> Studio:
    return Studio(project_root, DEFAULT_DB, DEFAULT_MANAGED)


def fixtures_dir() -> Path:
    return _PKG_DIR / "fixtures"


# ---------------------------------------------------------------------------
# Sidebar : projet + scan + exports
# ---------------------------------------------------------------------------

st.sidebar.title("🛠️ Rule Studio")
st.sidebar.caption("Gouvernance des règles du pipeline vSense / WYSIWYG")

project_root = st.sidebar.text_input("Racine du projet", value=str(_REPO_ROOT))
studio = get_studio(project_root)

col_a, col_b = st.sidebar.columns(2)
if col_a.button("🔍 Scanner", width="stretch"):
    with st.spinner("Scan en cours…"):
        stats = studio.scan(with_usage=True)
    st.sidebar.success(
        f"{stats['total']} règles "
        f"(+{stats['added']} / maj {stats['updated']} / "
        f"éditions préservées {stats['preserved']})"
    )
if col_b.button("♻️ Rafraîchir", width="stretch"):
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Exports")
records = studio.rules()
if records:
    st.sidebar.download_button(
        "⬇️ CSV", studio.export_csv(), "rules.csv", "text/csv", width="stretch"
    )
    st.sidebar.download_button(
        "⬇️ JSON", studio.export_json(), "rules.json", "application/json", width="stretch"
    )
    st.sidebar.download_button(
        "⬇️ Markdown", studio.export_markdown(), "rules.md", "text/markdown", width="stretch"
    )
    if st.sidebar.button("📄 Générer rapport d'audit", width="stretch"):
        report = studio.export_audit()
        out = _REPO_ROOT / "RULE_AUDIT_REPORT.md"
        out.write_text(report, encoding="utf-8")
        st.sidebar.success(f"Rapport écrit : {out}")
        st.sidebar.download_button(
            "⬇️ Télécharger le rapport", report, "RULE_AUDIT_REPORT.md", "text/markdown"
        )

# ---------------------------------------------------------------------------
# Corps principal
# ---------------------------------------------------------------------------

st.title("Console de gouvernance des règles")

if not records:
    st.info("Aucune règle en base. Cliquez sur **Scanner** dans la barre latérale.")
    st.stop()

tab_table, tab_new, tab_sim, tab_patch, tab_audit = st.tabs(
    ["📋 Inventaire", "➕ Nouvelle règle", "🧪 Simuler", "🩹 Diff & patch", "🔬 Audits pipeline"]
)

# ===========================================================================
# Onglet Inventaire
# ===========================================================================
with tab_table:
    st.subheader("Inventaire des règles")

    # --- Filtres ----------------------------------------------------------
    with st.expander("Filtres", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        units = ["(toutes)"] + sorted({r.pipeline_unit for r in records})
        types = ["(tous)"] + sorted({r.rule_type for r in records})
        statuses = ["(tous)"] + sorted({r.status for r in records})
        usages = ["(tous)"] + sorted({r.usage_status for r in records})
        sel_unit = f1.selectbox("Unité pipeline", units)
        sel_type = f2.selectbox("Type de règle", types)
        sel_status = f3.selectbox("Statut", statuses)
        sel_usage = f4.selectbox("Usage", usages)

        g1, g2, g3, g4 = st.columns(4)
        sources = ["(toutes)"] + sorted({r.source_type for r in records})
        risks = ["(tous)"] + sorted({r.risk_level for r in records})
        sel_source = g1.selectbox("Source", sources)
        sel_risk = g2.selectbox("Risque", risks)
        min_conf = g3.slider("Confiance min.", 0.0, 1.0, 0.0, 0.05)
        text_filter = g4.text_input("Filtre texte (fichier/titre)")

    def keep(r: RuleRecord) -> bool:
        if sel_unit != "(toutes)" and r.pipeline_unit != sel_unit:
            return False
        if sel_type != "(tous)" and r.rule_type != sel_type:
            return False
        if sel_status != "(tous)" and r.status != sel_status:
            return False
        if sel_usage != "(tous)" and r.usage_status != sel_usage:
            return False
        if sel_source != "(toutes)" and r.source_type != sel_source:
            return False
        if sel_risk != "(tous)" and r.risk_level != sel_risk:
            return False
        if r.confidence < min_conf:
            return False
        if text_filter:
            blob = f"{r.file_path} {r.title} {r.symbol_name}".lower()
            if text_filter.lower() not in blob:
                return False
        return True

    filtered = [r for r in records if keep(r)]
    st.caption(f"{len(filtered)} / {len(records)} règles affichées")

    # --- Tableau éditable -------------------------------------------------
    rows = []
    for r in filtered:
        row = r.to_row()
        row["_select"] = False
        rows.append({k: row.get(k) for k in (["_select"] + DISPLAY_COLS)})

    _BOOL_LABELS = {
        "keep": "Garder", "active": "Activée", "selected": "Sél.", "modify": "Modifier",
        "delete": "Supprimer", "validate": "Valider", "simulate": "Simuler",
        "implement": "Implémenter", "blocking": "Bloquante", "requires_tests": "Tests requis",
        "apply_to_project": "Appliquer",
    }
    _NL_LABELS = {
        "natural_language_title": "Titre naturel", "natural_language_rule": "Règle (langage naturel)",
        "objective": "Objectif", "understanding": "Compréhension",
        "human_decision": "Décision humaine", "proposed_modification": "Modification proposée",
    }
    col_config = {
        "_select": st.column_config.CheckboxColumn("✓", width="small"),
        "id": st.column_config.TextColumn("ID", disabled=True),
        "pipeline_unit": st.column_config.TextColumn("Unité", disabled=True),
        "rule_type": st.column_config.TextColumn("Type", disabled=True),
        "file_path": st.column_config.TextColumn("Fichier", disabled=True),
        "line_start": st.column_config.NumberColumn("Ligne", disabled=True),
        "source_type": st.column_config.TextColumn("Source", disabled=True),
        "managed": st.column_config.CheckboxColumn("Géré", disabled=True),
        "ai_interpreted": st.column_config.CheckboxColumn("IA", disabled=True),
        "confidence": st.column_config.NumberColumn("Conf.", disabled=True, format="%.2f"),
        "usage_status": st.column_config.TextColumn("Usage", disabled=True),
        "recommended_decision": st.column_config.SelectboxColumn(
            "Décision conseillée", options=KEEP_DECISIONS),
        "lifecycle_status": st.column_config.SelectboxColumn(
            "Cycle de vie", options=LIFECYCLE_STATUSES),
        "risk_level": st.column_config.SelectboxColumn("Risque", options=RISK_LEVELS),
        "validation_status": st.column_config.SelectboxColumn(
            "Valide ?", options=VALIDATION_STATUSES),
    }
    # Cases à cocher de décision (vrais booléens, directive §8.1).
    for b in BOOL_DECISION_FIELDS:
        col_config[b] = st.column_config.CheckboxColumn(_BOOL_LABELS.get(b, b))
    # Colonnes texte naturelles éditables.
    for c, label in _NL_LABELS.items():
        col_config[c] = st.column_config.TextColumn(label)
    # Reste : texte non éditable.
    for c in DISPLAY_COLS:
        if c not in EDITABLE_COLS and c not in col_config:
            col_config[c] = st.column_config.TextColumn(c, disabled=True)

    edited = st.data_editor(
        rows,
        column_config=col_config,
        width="stretch",
        height=460,
        hide_index=True,
        key="rule_table",
    )

    selected_ids = [row["id"] for row in edited if row.get("_select")]

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("💾 Enregistrer les éditions", type="primary"):
        n = 0
        triggered = {"interpret": 0, "patch": 0, "validate": 0}
        by_id = {r.id: r for r in records}
        for row in edited:
            rec = by_id.get(row["id"])
            if not rec:
                continue
            changed = False
            for col in EDITABLE_COLS:
                new_val = row.get(col)
                if new_val is not None and new_val != getattr(rec, col):
                    setattr(rec, col, new_val)
                    changed = True
            if changed:
                rec.lifecycle_status = "edited_by_human"
                studio.save(rec)
                n += 1
                # workflow §10 : ré-interpréter si naturel incomplet
                if not rec.natural_language_rule:
                    studio.interpret_rules([rec.id], only_uninterpreted=False)
                    triggered["interpret"] += 1
                # implement=true -> patch PROPOSÉ (jamais appliqué directement)
                if rec.implement:
                    studio.generate_patch(rec.id)
                    triggered["patch"] += 1
                if rec.validate:
                    studio.validate_rule(rec.id)
                    triggered["validate"] += 1
        st.success(f"{n} règle(s) mise(s) à jour. "
                   f"IA: {triggered['interpret']} interprétées, "
                   f"{triggered['patch']} patchs proposés, {triggered['validate']} validées.")
        if any(triggered.values()):
            st.info("Patchs proposés visibles dans l'onglet **Diff & patch** — "
                    "aucune application silencieuse.")
        st.rerun()

    if c4.button("🤖 Interpréter avec IA"):
        ids = selected_ids or None
        with st.spinner("Interprétation…"):
            summ = studio.interpret_rules(ids, only_uninterpreted=not bool(selected_ids))
        st.success(
            f"{summ['interpreted']} règle(s) interprétée(s) — "
            f"{summ['needs_human_review']} à revoir, "
            f"{summ['critical_units']} critiques (traduction/reconstruction).")
        st.rerun()
    if c2.button("🗑️ Supprimer la sélection"):
        deleted = 0
        for rid in selected_ids:
            rec = studio.get(rid)
            if rec and rec.deletable:
                studio.delete(rid)
                deleted += 1
        st.success(f"{deleted} règle(s) supprimée(s) ({len(selected_ids) - deleted} protégées).")
        st.rerun()

    if c3.button("⭐ Promouvoir en règle gérée"):
        n = 0
        for rid in selected_ids:
            rec = studio.get(rid)
            if rec and not rec.managed:
                rec.managed = True
                rec.simulable = bool(rec.dsl.get("when"))
                studio.save(rec)
                n += 1
        st.success(f"{n} règle(s) promue(s) en gérée(s).")
        st.rerun()

    if selected_ids:
        st.session_state["selected_ids"] = selected_ids
        st.caption(f"Sélection courante : {', '.join(selected_ids)}")

# ===========================================================================
# Onglet Nouvelle règle
# ===========================================================================
with tab_new:
    st.subheader("Créer une nouvelle règle gérée")
    with st.form("new_rule"):
        c1, c2, c3 = st.columns(3)
        title = c1.text_input("Titre")
        unit = c2.selectbox("Unité pipeline", PIPELINE_UNITS)
        rtype = c3.selectbox("Type de règle", RULE_TYPES)

        objective = st.text_area("Objectif", height=68)
        understanding = st.text_area("Compréhension", height=68)

        c4, c5, c6 = st.columns(3)
        risk = c4.selectbox("Risque", RISK_LEVELS, index=1)
        status = c5.selectbox("Statut initial", STATUSES, index=2)
        tests = c6.text_input("Tests à créer (séparés par des virgules)")

        c7, c8 = st.columns(2)
        target_file = c7.text_input("Fichier cible")
        target_fn = c8.text_input("Fonction cible")

        st.markdown("**Condition WHEN** (JSON DSL)")
        when_str = st.text_area(
            "when",
            value='{"all": [{"field": "page.index", "op": "eq", "value": 0}]}',
            height=110,
            label_visibility="collapsed",
        )
        st.markdown("**Action THEN** (JSON DSL)")
        then_str = st.text_area(
            "then",
            value='{"set": {"page.role": "cover"}}',
            height=90,
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button("Créer (preview)")

    if submitted:
        try:
            when = json.loads(when_str) if when_str.strip() else {}
            then = json.loads(then_str) if then_str.strip() else {}
        except json.JSONDecodeError as exc:
            st.error(f"JSON invalide : {exc}")
        else:
            dsl = {"when": when, "then": then}
            errors = validate_dsl(dsl)
            if errors:
                st.error("DSL invalide :\n- " + "\n- ".join(errors))
            else:
                rid = make_rule_id("MGD", f"{unit}:{title}")
                rec = RuleRecord(
                    id=rid,
                    title=title or rid,
                    pipeline_unit=unit,
                    rule_type=rtype,
                    source_type="manual",
                    objective=objective,
                    understanding=understanding,
                    status=status,
                    risk_level=risk,
                    managed=True,
                    simulable=bool(when),
                    validation_status="needs_review",
                    confidence=1.0,
                    tests_linked=[t.strip() for t in tests.split(",") if t.strip()],
                    dsl=dsl,
                    target={"file": target_file, "function": target_fn},
                    lifecycle_status="candidate",
                    natural_language_title=title or rid,
                    natural_language_rule=objective or understanding,
                    ai_interpreted=False,
                    coding_agent_status="not_started",
                )
                rec.file_path = target_file
                studio.save(rec)
                st.success(
                    f"Règle `{rid}` créée (statut **candidate**) et écrite dans "
                    "managed_rules.yaml. Aucune implémentation immédiate."
                )

# ===========================================================================
# Onglet Simuler
# ===========================================================================
with tab_sim:
    st.subheader("Simuler une règle sur un contexte")
    simulable = [r for r in records if r.simulable and r.dsl.get("when")]
    if not simulable:
        st.info("Aucune règle simulable (il faut un DSL `when`). Créez ou promouvez une règle gérée.")
    else:
        labels = {f"{r.id} — {r.title[:60]}": r.id for r in simulable}
        choice = st.selectbox("Règle", list(labels.keys()))
        rec = studio.get(labels[choice])

        st.code(json.dumps(rec.dsl, ensure_ascii=False, indent=2), language="json")

        fixtures = sorted(fixtures_dir().glob("*.json"))
        fx_names = ["(saisie manuelle)"] + [f.name for f in fixtures]
        fx_choice = st.selectbox("Contexte (fixture)", fx_names)
        if fx_choice != "(saisie manuelle)":
            ctx_str = (fixtures_dir() / fx_choice).read_text(encoding="utf-8")
        else:
            ctx_str = '{\n  "page": {"index": 0, "visual_density": 0.8, "text_density": 0.1, "image_count": 1}\n}'
        ctx_str = st.text_area("Contexte JSON", value=ctx_str, height=220)

        if st.button("▶️ Lancer la simulation", type="primary"):
            try:
                ctx = json.loads(ctx_str)
            except json.JSONDecodeError as exc:
                st.error(f"Contexte JSON invalide : {exc}")
            else:
                res = simulate(rec.dsl, ctx)
                from tools.rule_studio.core.usage_analyzer import record_hit

                record_hit(rec.id, rec.pipeline_unit, res.matched)

                if res.matched:
                    st.success("✅ Condition VRAIE — la règle se déclenche.")
                else:
                    st.warning("⛔ Condition FAUSSE — la règle ne se déclenche pas.")

                cc1, cc2 = st.columns(2)
                with cc1:
                    st.markdown("**Champs lus**")
                    st.dataframe(res.reads, width="stretch")
                    st.markdown("**Trace**")
                    st.code("\n".join(res.trace) or "(aucune)")
                with cc2:
                    st.markdown("**Actions appliquées**")
                    st.dataframe(res.actions or [{"info": "aucune"}], width="stretch")
                    if res.warnings:
                        for w in res.warnings:
                            st.warning(w)

                st.markdown("**Avant / Après**")
                d1, d2 = st.columns(2)
                d1.json(res.before)
                d2.json(res.after)

# ===========================================================================
# Onglet Diff & patch
# ===========================================================================
with tab_patch:
    st.subheader("Générer un diff et appliquer (sécurisé)")
    st.caption(
        "V1 sûre : insertion d'un bloc commentaire de règle dans le fichier cible. "
        "Jamais d'application silencieuse — backup Git + confirmation requise."
    )
    patchable = [r for r in records if (r.target.get("file") or r.file_path)]
    if not patchable:
        st.info("Aucune règle avec fichier cible.")
    else:
        labels = {f"{r.id} — {r.title[:60]}": r.id for r in patchable}
        choice = st.selectbox("Règle à patcher", list(labels.keys()), key="patch_sel")
        rec = studio.get(labels[choice])

        with st.expander("🤖 Patch proposé par l'agent de codage (jamais appliqué)", expanded=False):
            if st.button("Générer un patch (agent IA)"):
                out = studio.generate_patch(rec.id) or {}
                st.session_state["agent_patch"] = out
            ap = st.session_state.get("agent_patch")
            if ap:
                st.write(f"**Statut** : `{ap.get('status')}` · risque `{ap.get('risk_level')}`")
                st.write(f"Fichiers cibles : {', '.join(ap.get('target_files') or []) or '—'}")
                if ap.get("tests_to_run"):
                    st.write("Tests : " + ", ".join(ap["tests_to_run"]))
                if ap.get("patch"):
                    st.code(ap["patch"], language="diff")
                st.caption(ap.get("explanation") or "")
                st.warning("Proposition uniquement — application contrôlée requise (diff + tests + confirmation).")

        if st.button("🔎 Générer le diff"):
            prop = patcher.propose_comment_patch(rec, project_root)
            st.session_state["patch_prop"] = prop.__dict__

        prop_d = st.session_state.get("patch_prop")
        if prop_d and prop_d["rule_id"] == rec.id:
            if not prop_d["safe"]:
                st.error(f"⚠️ {prop_d['reason']}")
            if prop_d["diff"]:
                st.code(prop_d["diff"], language="diff")
                confirm = st.checkbox("Je confirme l'application de ce patch")
                make_backup = st.checkbox("Créer une branche de backup Git", value=True)
                ap1, ap2 = st.columns(2)
                if ap1.button("✅ Appliquer", disabled=not (confirm and prop_d["safe"])):
                    prop = patcher.PatchProposal(**prop_d)
                    prop = patcher.apply_patch(prop, project_root, confirm=confirm, make_backup=make_backup)
                    if prop.applied:
                        st.success(f"Patch appliqué. Backup : {prop.backup_branch or '(aucun)'}")
                        st.session_state["last_applied"] = prop.__dict__
                    for w in prop.warnings:
                        st.warning(w)
                if ap2.button("🧪 Lancer les tests liés"):
                    with st.spinner("pytest…"):
                        tr = run_tests(project_root, rec.tests_linked or None)
                    (st.success if tr.ok else st.error)(tr.summary)
                    st.code(tr.output[-3000:])
            else:
                st.info(prop_d["reason"] or "Aucun changement.")

        last = st.session_state.get("last_applied")
        if last and st.button("↩️ Rollback du dernier patch"):
            prop = patcher.PatchProposal(**last)
            if patcher.rollback(prop, project_root):
                st.success("Rollback effectué.")
                st.session_state.pop("last_applied", None)

# ===========================================================================
# Onglet Audits pipeline (les vrais audits PROUVENT — l'IA explique)
# ===========================================================================
with tab_audit:
    st.subheader("Statut des vrais audits du pipeline")
    st.caption(
        "Rule Studio explique et propose ; ces audits réels (PAGETRANSLATE / "
        "PAGERECONSTRUCT / PUBREADY) PROUVENT. Disponibilité ci-dessous ; le statut "
        "détaillé se calcule lors d'un run avec un plan + normalized."
    )
    from tools.rule_studio.core.pipeline_audit_bridge import (  # noqa: E402
        AUDIT_TARGETS, audit_availability)

    avail = audit_availability()
    by_id = {r.id: r for r in records}
    table = []
    for rid, info in avail.items():
        rec = by_id.get(rid)
        table.append({
            "Règle": rid,
            "Bloquante": bool(getattr(rec, "blocking", False)) if rec else False,
            "Audit": f"{info['module']}.{info['function']}",
            "Disponible": info["available"],
            "Erreur": info["error"] or "",
            "Statut validation": getattr(rec, "validation_status", "—") if rec else "(règle absente)",
        })
    st.dataframe(table, width="stretch", hide_index=True)
    missing = [rid for rid, i in avail.items() if not i["available"]]
    if missing:
        st.warning("Audits non importables : " + ", ".join(missing))
    else:
        st.success("Tous les audits de couverture liés sont importables.")
