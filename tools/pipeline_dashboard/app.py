#!/usr/bin/env python3
"""Streamlit dashboard for the PAGEPRINT -> PAGETRANSLATE pipeline.

A dense, editable TABLE (not cards) to monitor and verify that extraction /
selection rules are correctly applied, page by page:

  - one row = one granular element (phrase / expression / word / abbreviation,
    or a non-textual zone: formula / code / table / figure / region);
  - columns: granularity, role, type, translatable, source, translation,
    status, length-ratio, qa flags, bbox;
  - multiline text wraps in the cells; translatable / translation / role are
    editable inline and saved back to SQLite;
  - a "règles de l'art" KPI banner; horizontal filters; the page image
    (bboxes / source / background) alongside.

Run:
    streamlit run tools/pipeline_dashboard/app.py [-- --db <path.db>]
"""

from __future__ import annotations

import glob
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import json
import re

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LIVE_DIR = ROOT / "results" / "dashboard_live"

CAT_TAG = {"texte": "🟩 texte", "texte_exclu": "🟥 exclu", "formule": "🟨 formule",
           "code": "🟧 code", "table": "🟦 table", "figure": "🟪 figure", "region": "⬜ région"}
CAT_COLOR = {"texte": "#E8F5E9", "texte_exclu": "#FDECEA", "formule": "#FFF8E1",
             "code": "#FFF0E6", "table": "#E7F0FB", "figure": "#F3E8FB", "region": "#F1F1F1"}
_TAG_COLOR = {CAT_TAG[k]: CAT_COLOR[k] for k in CAT_TAG}


def _style_view(view):
    def color_row(row):
        bg = _TAG_COLOR.get(row.get("cat"), "#FFFFFF")
        return [f"background-color: {bg}"] * len(row)
    return view.style.apply(color_row, axis=1)
ROLE_OPTIONS = [
    "", "title", "section_heading", "subsection_heading", "body_paragraph", "list_item",
    "figure_caption", "table_caption", "table_header_cell", "table_body_cell",
    "formula_expression", "formula_explanation", "code_line", "command_name", "path",
    "page_header", "page_footer", "publisher_mark", "author_name", "author_bio",
    "index_entry", "index_head_term", "index_subentry", "index_page_reference",
    "toc_entry_title", "toc_page_reference", "diagram_label", "unknown",
]


# ----------------------------------------------------------------- DB helpers
def load_pages(db: str) -> pd.DataFrame:
    con = sqlite3.connect(db)
    try:
        return pd.read_sql_query("SELECT * FROM pages ORDER BY page_tag", con)
    finally:
        con.close()


def load_elements(db: str, page_tag: str) -> pd.DataFrame:
    con = sqlite3.connect(db)
    try:
        df = pd.read_sql_query("SELECT * FROM elements WHERE page_tag=? ORDER BY ord", con, params=(page_tag,))
    finally:
        con.close()
    if not df.empty:
        df["translatable"] = df["translatable"].astype(bool)
        df["needs_review"] = df["needs_review"].astype(bool)
        df["edited"] = df["edited"].astype(bool)
    return df


def save_changes(db: str, original: pd.DataFrame, edited: pd.DataFrame) -> int:
    con = sqlite3.connect(db)
    now = datetime.now().isoformat(timespec="seconds")
    orig = original.set_index("id")
    changed = 0
    for _, row in edited.iterrows():
        rid = int(row["id"])
        if rid not in orig.index:
            continue
        o = orig.loc[rid]
        if (bool(row["translatable"]) != bool(o["translatable"])
                or str(row["translation"]) != str(o["translation"])
                or str(row["role"]) != str(o["role"])):
            con.execute(
                "UPDATE elements SET translatable=?, translation=?, role=?, edited=1, updated_at=? WHERE id=?",
                (int(bool(row["translatable"])), row["translation"], row["role"], now, rid),
            )
            changed += 1
    con.commit()
    con.close()
    return changed


def _arg_db() -> str | None:
    if "--db" in sys.argv:
        i = sys.argv.index("--db")
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def _find_dbs() -> list[str]:
    return sorted(glob.glob(str(ROOT / "results" / "**" / "*.db"), recursive=True), reverse=True)


# ----------------------------------------------------------------- live run
@st.cache_resource(show_spinner=False)
def get_orchestrator():
    from tools.run_pageprint_pagetranslate_audit import make_orchestrator
    (LIVE_DIR / "source_pages").mkdir(parents=True, exist_ok=True)
    return make_orchestrator(str(LIVE_DIR / "source_pages"))


@st.cache_resource(show_spinner=False)
def get_engine(engine: str, model: str):
    from tools.run_pageprint_pagetranslate_audit import make_engine
    return make_engine(engine, model=model)


def run_pipeline_live(pdf: str, page: int, engine: str, model: str) -> dict:
    from tools.run_pageprint_pagetranslate_audit import run_page
    from tools.pipeline_dashboard.ingest import ingest_dir
    metrics = run_page(get_orchestrator(), get_engine(engine, model), Path(pdf), page, LIVE_DIR)
    if "error" in metrics:
        return metrics
    metrics["db"] = ingest_dir(LIVE_DIR, db_path=LIVE_DIR / "pipeline_dashboard.db")["db_path"]
    return metrics


# ----------------------------------------------------------------- table prep
def _ratio(src: str, tr: str) -> float:
    src = str(src or ""); tr = str(tr or "")
    if not src:
        return 0.0
    return round(len(tr) / max(1, len(src)), 2)


def _qa_flags(row) -> str:
    flags = []
    qa = str(row["qa_reasons"] or "")
    if "repeated_output" in qa:
        flags.append("🔁")
    if "source_fragment" in qa or "dehyphenation_needed" in qa:
        flags.append("✂️")
    if "number_mismatch" in qa:
        flags.append("🔢")
    if "protected_token_mismatch" in qa:
        flags.append("🔒")
    if "source_leak" in qa or "unchanged_suspect" in qa:
        flags.append("🅰️")
    if row["category"] in {"texte", "texte_exclu"} and row["translatable"] and not str(row["translation"]).strip():
        flags.append("∅")
    return " ".join(flags)


def to_view(df: pd.DataFrame) -> pd.DataFrame:
    v = pd.DataFrame()
    v["id"] = df["id"]
    v["#"] = df["ord"]
    v["cat"] = df["category"].map(lambda c: CAT_TAG.get(c, c))
    v["gran"] = df["granularity"]
    v["traduisible"] = df["translatable"]
    v["rôle"] = df["role"].fillna("")
    v["type"] = df["object_type"].fillna("")
    v["source"] = df["source_text"].fillna("")
    v["traduction"] = df["translation"].fillna("")
    v["statut"] = df["status"].fillna("")
    v["Δ"] = [_ratio(s, t) for s, t in zip(df["source_text"], df["translation"])]
    v["qa"] = [_qa_flags(r) for _, r in df.iterrows()]
    v["✏️"] = df["edited"]
    v["bbox"] = df["bbox"].fillna("")
    return v


def from_view(view: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    out = original.set_index("id").copy()
    for _, r in view.iterrows():
        rid = int(r["id"])
        if rid in out.index:
            out.at[rid, "translatable"] = bool(r["traduisible"])
            out.at[rid, "translation"] = r["traduction"]
            out.at[rid, "role"] = r["rôle"]
    return out.reset_index()


def kpis(df: pd.DataFrame) -> dict:
    txt = df[df["category"].isin(["texte", "texte_exclu"])]
    transl = df[(df["category"] == "texte") & (df["translatable"])]
    n_tr = len(transl)
    phrases = int((transl["granularity"] == "phrase").sum())
    role_ok = int((txt["role"].fillna("").astype(str).str.len() > 0).sum())
    publisher = int(((df["role"] == "publisher_mark") & (df["translatable"])).sum())
    empty = int(((df["category"] == "texte") & (df["translatable"]) & (df["translation"].fillna("").str.strip() == "")).sum())
    return {
        "traduisibles": n_tr,
        "% phrases": f"{round(100 * phrases / n_tr)}%" if n_tr else "—",
        "% rôles posés": f"{round(100 * role_ok / len(txt))}%" if len(txt) else "—",
        "à revoir": int(df["needs_review"].sum()),
        "publisher envoyés": publisher,
        "trad. vides": empty,
    }


# ----------------------------------------------------------------- zones & tables
def _norm(t) -> str:
    return re.sub(r"\s+", " ", str(t or "")).strip().lower()


@st.cache_data(show_spinner=False)
def load_input_data(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _geom_scale(input_data: dict):
    g = (input_data.get("page") or {}).get("geometry") or {}
    sx, sy = g.get("scale_x_px_per_pt"), g.get("scale_y_px_per_pt")
    return (float(sx), float(sy)) if sx and sy else (1.0, 1.0)


def crop_region(img: Image.Image, bbox, sx: float, sy: float):
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None
    x0, y0, x1, y1 = bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy
    x0, y0 = max(0, x0 - 2), max(0, y0 - 2)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    return img.crop((x0, y0, min(img.width, x1 + 2), min(img.height, y1 + 2)))


def _parse_bbox(val):
    try:
        b = json.loads(str(val))
        if isinstance(b, list) and len(b) == 4:
            return [float(x) for x in b]
    except Exception:
        pass
    return None


def _draw_wrapped(draw, text, x0, y0, x1, y1, font):
    max_w = max(10, x1 - x0)
    words = str(text).split()
    line, y = "", y0
    line_h = (font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) + 3 if hasattr(font, "getbbox") else 14
    for w in words:
        trial = (line + " " + w).strip()
        width = draw.textlength(trial, font=font) if hasattr(draw, "textlength") else len(trial) * 7
        if width > max_w and line:
            draw.text((x0, y), line, fill=(20, 20, 20), font=font)
            y += line_h
            line = w
        else:
            line = trial
    if line:
        draw.text((x0, y), line, fill=(20, 20, 20), font=font)


def _containing_block(units, bbox, levels):
    """Tightest unit of the given levels whose box contains the element centre."""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    best, best_area = None, None
    for u in units:
        if u.get("level") not in levels:
            continue
        b = (u.get("geometry") or {}).get("bbox")
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        if b[0] - 1 <= cx <= b[2] + 1 and b[1] - 1 <= cy <= b[3] + 1:
            area = (b[2] - b[0]) * (b[3] - b[1])
            if best_area is None or area < best_area:
                best, best_area = [float(x) for x in b], area
    return best


def render_element_on_blank(input_data: dict, source_img, row) -> Image.Image:
    """Full-page canvas: the parent block at its real position + element highlighted.

    The page-sized canvas keeps every element at its true scale (no zoom): the
    block is pasted where it really sits, and the selected element is outlined.
    """
    sx, sy = _geom_scale(input_data)
    geom = (input_data.get("page") or {}).get("geometry") or {}
    W = int(geom.get("render_width_px") or (source_img.width if source_img else 850))
    H = int(geom.get("render_height_px") or (source_img.height if source_img else 1100))
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    bbox = _parse_bbox(row.get("bbox"))
    if not bbox:
        ImageDraw.Draw(canvas).text((20, 20), "(position inconnue — pas de bbox)", fill=(160, 0, 0))
        return canvas
    levels = {"table", "region", "block"} if row.get("category") in {"table", "figure", "formule", "code", "region"} else {"block"}
    block = _containing_block(input_data.get("units") or [], bbox, levels) or bbox
    # Paste the block content from the source page at its real position.
    if source_img is not None:
        crop = crop_region(source_img, block, sx, sy)
        if crop is not None:
            canvas.paste(crop, (int(max(0, block[0] * sx)), int(max(0, block[1] * sy))))
    draw = ImageDraw.Draw(canvas, "RGBA")
    bx = [block[0] * sx, block[1] * sy, block[2] * sx, block[3] * sy]
    draw.rectangle(bx, outline=(150, 150, 150), width=1)
    ex = [bbox[0] * sx - 2, bbox[1] * sy - 2, bbox[2] * sx + 2, bbox[3] * sy + 2]
    draw.rectangle(ex, fill=(255, 215, 0, 80))
    draw.rectangle(ex, outline=(220, 20, 60), width=3)
    return canvas


def trans_map(df: pd.DataFrame) -> dict:
    m = {}
    for _, r in df[df["category"] == "texte"].iterrows():
        m[_norm(r["source_text"])] = r["translation"]
    return m


def build_table_grids(table: dict, tmap: dict):
    cells = table.get("cells") or []
    rows: dict = {}
    for c in cells:
        rows.setdefault(c.get("row_index") or 0, []).append(c)
    src, tgt = [], []
    for ri in sorted(rows):
        line = sorted(rows[ri], key=lambda c: (c.get("bbox") or [0])[0])
        src.append([str(c.get("text") or "") for c in line])
        tgt.append([
            ("⟦préservé⟧" if c.get("translation_mode") == "preserve_text_exactly"
             else tmap.get(_norm(c.get("text")), ""))
            for c in line
        ])
    width = max((len(r) for r in src), default=0)
    pad = lambda g: [r + [""] * (width - len(r)) for r in g]
    cols = [f"col {i+1}" for i in range(width)]
    return pd.DataFrame(pad(src), columns=cols), pd.DataFrame(pad(tgt), columns=cols)


def render_zones_and_tables(prow, df: pd.DataFrame) -> None:
    path = prow["input_data_path"] if "input_data_path" in prow.index else None
    if not path or not Path(path).is_file():
        st.info("input_data indisponible pour cette page (ré-ingère la base).")
        return
    data = load_input_data(path)
    ls = data.get("logical_structures") or {}
    src_img_path = prow["source_image"]
    img = Image.open(src_img_path).convert("RGB") if src_img_path and Path(src_img_path).is_file() else None
    sx, sy = _geom_scale(data)
    tmap = trans_map(df)

    # ---- Tables (contenu avant / après) ----
    tables = ls.get("tables") or []
    st.markdown(f"### 📊 Tables extraites ({len(tables)})")
    for t in tables:
        st.markdown(f"**{t.get('table_id')}** · {len(t.get('cells') or [])} cellules · "
                    f"stratégie: {t.get('detection_strategy', '—')}")
        if img is not None:
            crop = crop_region(img, t.get("bbox"), sx, sy)
            if crop is not None:
                st.image(crop, caption="zone table (source)", width="stretch")
        src_grid, tgt_grid = build_table_grids(t, tmap)
        ca, cb = st.columns(2)
        ca.caption("Contenu source"); ca.dataframe(src_grid, width="stretch", hide_index=True)
        cb.caption("Contenu traduit"); cb.dataframe(tgt_grid, width="stretch", hide_index=True)
        st.divider()
    if not tables:
        st.caption("aucune table détectée")

    # ---- Zones préservées détectées (YOLO / glyphes) ----
    regions = (data.get("views") or {}).get("detected_regions") or data.get("regions") or []
    st.markdown(f"### 🧩 Zones spéciales préservées ({len(regions)})")
    by_type: dict = {}
    for r in regions:
        by_type.setdefault(r.get("object_type") or r.get("region_type") or "zone", []).append(r)
    for otype, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
        with st.expander(f"{otype} — {len(items)} zone(s)", expanded=(otype in {"formula", "code", "table"})):
            if img is None:
                st.caption("pas d'image source pour le crop")
                continue
            cols = st.columns(4)
            for i, r in enumerate(items[:24]):
                crop = crop_region(img, r.get("bbox"), sx, sy)
                with cols[i % 4]:
                    if crop is not None:
                        st.image(crop, width="stretch")
                    st.caption(f"{r.get('detection_source') or r.get('source') or '—'} · "
                               f"conf {r.get('confidence', '—')}")


# ----------------------------------------------------------------- main
def main() -> None:
    st.set_page_config(page_title="Suivi pipeline PAGEPRINT→PAGETRANSLATE", layout="wide")
    db = st.session_state.get("db") or _arg_db()

    with st.sidebar:
        st.title("🧭 Suivi pipeline")
        mode = st.radio("Mode", ["Parcourir une base", "▶️ Lancer le pipeline (live)"])
        if mode == "▶️ Lancer le pipeline (live)":
            uploaded = st.file_uploader("📂 Choisir un PDF (fenêtre)", type=["pdf"])
            pdf_dir = ROOT / "tests" / "doc_pdf"
            pdfs = sorted(str(p) for p in pdf_dir.glob("*.pdf")) if pdf_dir.is_dir() else []
            corpus = st.selectbox("…ou un PDF du corpus", ["—"] + [Path(p).name for p in pdfs]) if pdfs else "—"
            pdf = None
            if uploaded is not None:
                up = ROOT / "results" / "uploads" / "dashboard_uploads"; up.mkdir(parents=True, exist_ok=True)
                pdf = str(up / uploaded.name); Path(pdf).write_bytes(uploaded.getvalue())
            elif corpus and corpus != "—":
                pdf = str(pdf_dir / corpus)
            page_num = st.number_input("Page", min_value=1, value=1, step=1)
            engine = st.selectbox("Moteur", ["ct2", "mock", "rule"])
            model = st.text_input("Modèle", value="opus_mt_tc_big_en_fr")
            if st.button("▶️ Lancer", type="primary", disabled=pdf is None):
                with st.spinner(f"pipeline sur {Path(pdf).name} p{page_num}…"):
                    res = run_pipeline_live(pdf, int(page_num), engine, model)
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.session_state["db"] = res["db"]; st.session_state["page"] = res["tag"]
                    st.rerun()
        else:
            dbs = _find_dbs()
            if dbs:
                db = st.selectbox("Base SQLite", dbs, index=dbs.index(db) if db in dbs else 0)

    if not db or not Path(db).is_file():
        st.info("Choisis une base, ou bascule en **Lancer le pipeline**.")
        st.stop()
    pages = load_pages(db)
    if pages.empty:
        st.warning("Base vide."); st.stop()

    tags = pages["page_tag"].tolist()
    idx = tags.index(st.session_state["page"]) if st.session_state.get("page") in tags else 0

    # ---- top bar: page select + KPIs
    top = st.columns([3, 1, 1, 1, 1, 1, 1])
    page_tag = top[0].selectbox("Page", tags, index=idx)
    prow = pages[pages["page_tag"] == page_tag].iloc[0]
    df = load_elements(db, page_tag)
    k = kpis(df)
    top[1].metric("traduisibles", k["traduisibles"])
    top[2].metric("% phrases", k["% phrases"])
    top[3].metric("% rôles", k["% rôles posés"])
    top[4].metric("à revoir", k["à revoir"])
    top[5].metric("publisher⚠", k["publisher envoyés"])
    top[6].metric("trad. vides", k["trad. vides"])
    st.caption(f"page_role: **{prow['page_role']}** · santé: **{prow['selection_health']}** · "
               f"runtime: {prow['runtime_status']} · qualité: {prow['quality_status']} · publication: {prow['publication_status']}")

    # ---- horizontal filters
    f = st.columns([2, 1, 2, 2, 1])
    cats = f[0].multiselect("catégorie", list(CAT_TAG), default=[], placeholder="toutes")
    trad = f[1].radio("traduisible", ["tous", "oui", "non"], horizontal=True)
    grans = f[2].multiselect("granularité", ["phrase", "expression", "word", "abbreviation"], placeholder="toutes")
    search = f[3].text_input("recherche", placeholder="texte source / traduction")
    review_only = f[4].checkbox("à revoir")

    view_df = df.copy()
    if cats:
        view_df = view_df[view_df["category"].isin(cats)]
    if trad == "oui":
        view_df = view_df[view_df["translatable"]]
    elif trad == "non":
        view_df = view_df[~view_df["translatable"]]
    if grans:
        view_df = view_df[view_df["granularity"].isin(grans)]
    if search:
        s = search.lower()
        view_df = view_df[view_df["source_text"].str.lower().str.contains(s, na=False)
                          | view_df["translation"].str.lower().str.contains(s, na=False)]
    if review_only:
        view_df = view_df[view_df["needs_review"]]

    left, right = st.columns([5, 3], gap="medium")
    view = to_view(view_df)
    selected_rid = None

    with left:
        tab_table, tab_zones = st.tabs(["📋 Tableau des éléments", "🧩 Zones & Tables"])
        with tab_table:
            st.caption(f"{len(view_df)} ligne(s) — clique une ligne pour la sélectionner (aperçu + édition)")
            event = st.dataframe(
                _style_view(view.drop(columns=["id"])), width="stretch", hide_index=True, height=560,
                on_select="rerun", selection_mode="single-row", key=f"tbl_{page_tag}",
                column_config={
                    "source": st.column_config.TextColumn("source (PAGEPRINT)", width="large"),
                    "traduction": st.column_config.TextColumn("traduction (PAGETRANSLATE)", width="large"),
                },
            )
            rows = event.selection.rows if event and event.selection else []
            if rows:
                selected_rid = int(view.iloc[rows[0]]["id"])
            st.download_button("⬇️ CSV", view.to_csv(index=False).encode("utf-8"),
                               file_name=f"{page_tag}.csv", mime="text/csv")
            # Edit panel for the selected element.
            if selected_rid is not None:
                srow = df[df["id"] == selected_rid].iloc[0]
                role_options = sorted(set(ROLE_OPTIONS) | {str(srow["role"] or "")})
                with st.container(border=True):
                    st.markdown(f"**Édition #{int(srow['ord'])}** · {CAT_TAG.get(srow['category'], srow['category'])}")
                    e1, e2 = st.columns([1, 3])
                    new_tr = e1.checkbox("traduisible", value=bool(srow["translatable"]), key=f"e_tr_{selected_rid}")
                    new_role = e1.selectbox("rôle", role_options,
                                            index=role_options.index(str(srow["role"] or "")), key=f"e_role_{selected_rid}")
                    new_trad = e2.text_area("traduction", value=str(srow["translation"] or ""), key=f"e_trad_{selected_rid}", height=90)
                    e2.caption(f"source: {str(srow['source_text'])[:160]}")
                    if e1.button("💾 Enregistrer", type="primary", key=f"e_save_{selected_rid}"):
                        update_element(db, selected_rid, new_tr, new_trad, new_role)
                        st.success("enregistré.")
                        st.rerun()
        with tab_zones:
            render_zones_and_tables(prow, df)

    with right:
        tabs = st.tabs(["aperçu", "bboxes", "source", "fond"])
        with tabs[0]:
            if selected_rid is None:
                st.info("Clique une ligne du tableau pour voir l'élément dans son bloc, à sa position réelle.")
            else:
                row = df[df["id"] == selected_rid].iloc[0]
                ip = prow["input_data_path"] if "input_data_path" in prow.index else None
                data = load_input_data(ip) if ip and Path(str(ip)).is_file() else {}
                simg = prow["source_image"]
                src_img = Image.open(simg).convert("RGB") if simg and Path(simg).is_file() else None
                st.image(render_element_on_blank(data, src_img, row), width="stretch")
                st.caption(f"#{int(row['ord'])} · {row['category']} · bloc + élément surligné (taille réelle)")
        for tab, col, label in zip(tabs[1:], ["bboxes_image", "source_image", "background_image"], ["", "", "trame (texte masqué)"]):
            with tab:
                p = prow[col]
                if p and Path(p).is_file():
                    st.image(p, width="stretch")
                    if label:
                        st.caption(label)
                else:
                    st.info("indisponible")
        st.caption("qa: 🔁 répétition · ✂️ fragment/césure · 🔢 nombre · 🔒 token protégé · 🅰️ non traduit · ∅ vide")


if __name__ == "__main__":
    main()
