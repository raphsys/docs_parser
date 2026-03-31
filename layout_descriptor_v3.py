import math
import re


class LayoutDescriptorBuilderV3:
    VERSION = "layout_descriptor.v3"

    def build(self, page_data):
        if not isinstance(page_data, dict):
            return {}

        dims = page_data.get("dimensions") or {}
        width = float(dims.get("width", 0.0) or 0.0)
        height = float(dims.get("height", 0.0) or 0.0)
        columns = list(((page_data.get("layout") or {}).get("columns") or []))
        blocks = list(page_data.get("blocks") or [])
        page_number = int(page_data.get("page", 1) or 1)
        page_case_v2 = page_data.get("page_case_v2") or {}

        observed = self._build_observed_structure(page_data, blocks, columns, width, height)
        inferred = self._build_inferred_structure(page_data, observed, columns)
        synthetic = self._build_synthetic_structure(page_data, columns, width, height)
        hierarchy = self._build_hierarchy(observed, inferred, synthetic)
        dependency_graph = self._build_dependency_graph(page_data, observed, inferred)
        spatial_graph = self._build_spatial_graph(page_data, observed, inferred, columns)
        typographic_graph = self._build_typographic_graph(observed, inferred)
        primary_structure_family = self._determine_primary_structure_family(page_data, inferred, page_case_v2)
        structure_arbitration = self._build_structure_arbitration(primary_structure_family, inferred)
        render_model = self._build_render_model(
            page_data,
            observed,
            inferred,
            dependency_graph,
            spatial_graph,
            typographic_graph,
            structure_arbitration,
        )
        reconstruction_contract = self._build_reconstruction_contract(
            page_data,
            observed,
            inferred,
            synthetic,
            dependency_graph,
            spatial_graph,
            typographic_graph,
            render_model,
            page_case_v2,
            structure_arbitration,
        )

        return {
            "descriptor_version": self.VERSION,
            "page_id": page_number - 1,
            "page_number": page_number,
            "page_size": {"width": width, "height": height, "unit": "px"},
            "page_role": str(page_data.get("page_role") or "body").strip().lower(),
            "legacy_bridge": {
                "document_type": str(page_data.get("document_type") or "mixed_unknown"),
                "layout_type": str(page_data.get("layout_type") or "mixed_blocks"),
                "style_profile": str(page_data.get("style_profile") or "mixed_irregular"),
                "page_family": str(page_data.get("page_family") or "unknown"),
                "page_family_group": str(page_data.get("page_family_group") or "unknown"),
            },
            "classifier_bridge_v2": page_case_v2,
            "observed_structure": observed,
            "inferred_structure": inferred,
            "synthetic_structure": synthetic,
            "hierarchy": hierarchy,
            "dependency_graph": dependency_graph,
            "spatial_graph": spatial_graph,
            "typographic_graph": typographic_graph,
            "primary_structure_family": primary_structure_family,
            "structure_arbitration": structure_arbitration,
            "render_model": render_model,
            "reconstruction_contract": reconstruction_contract,
        }

    def _build_observed_structure(self, page_data, blocks, columns, page_w, page_h):
        observed_regions = []
        for col in columns:
            observed_regions.append(
                {
                    "id": f"obs_col_{int(col.get('id', len(observed_regions)) or len(observed_regions))}",
                    "type": "column",
                    "bbox": [
                        float(col.get("x0", 0.0) or 0.0),
                        0.0,
                        float(col.get("x1", page_w) or page_w),
                        page_h,
                    ],
                    "source": "observed_layout_columns",
                }
            )

        for idx, bbox in enumerate(page_data.get("non_text_zones") or []):
            norm = self._norm_bbox(bbox)
            if not norm:
                continue
            observed_regions.append(
                {
                    "id": f"obs_non_text_{idx}",
                    "type": "non_text_zone",
                    "bbox": norm,
                    "source": "observed_non_text",
                }
            )

        elements = []
        spans = []
        for block_idx, block in enumerate(blocks):
            block_id = str(block.get("id") or f"block_{block_idx}")
            bbox = self._norm_bbox(block.get("bbox"))
            if not bbox:
                continue
            role = str(block.get("role") or "body").strip().lower()
            block_node = {
                "id": block_id,
                "type": "block",
                "role": role,
                "bbox": bbox,
                "column_index": self._column_index_for_bbox(bbox, columns),
                "source": str(block.get("source") or "ocr"),
                "text": self._clean_text(block.get("translated_text") or block.get("text") or ""),
                "style": self._style_payload(block),
            }
            elements.append(block_node)
            for line_idx, line in enumerate(block.get("lines") or []):
                line_bbox = self._norm_bbox(line.get("bbox"))
                line_id = f"{block_id}::line::{line_idx}"
                line_text = self._clean_text(line.get("translated_text") or line.get("line_text") or "")
                line_node = {
                    "id": line_id,
                    "type": "line",
                    "role": role,
                    "parent_id": block_id,
                    "bbox": line_bbox or bbox,
                    "column_index": block_node["column_index"],
                    "source": block_node["source"],
                    "text": line_text,
                    "style": self._style_payload(line),
                }
                elements.append(line_node)
                for phrase_idx, phrase in enumerate(line.get("phrases") or []):
                    phrase_bbox = self._norm_bbox(phrase.get("bbox"))
                    phrase_text = self._clean_text(phrase.get("translated_text") or phrase.get("text") or phrase.get("texte") or "")
                    phrase_id = f"{line_id}::span::{phrase_idx}"
                    span_node = {
                        "id": phrase_id,
                        "type": "span",
                        "role": role,
                        "parent_id": line_id,
                        "bbox": phrase_bbox or line_node["bbox"],
                        "column_index": block_node["column_index"],
                        "source": block_node["source"],
                        "text": phrase_text,
                        "style": self._style_payload(phrase),
                    }
                    spans.append(span_node)

        return {
            "regions": observed_regions,
            "elements": elements,
            "spans": spans,
            "images": [
                {
                    "id": str(image.get("id") or f"image_{idx}"),
                    "bbox": self._norm_bbox(image.get("bbox")),
                    "source": "observed_image",
                }
                for idx, image in enumerate(page_data.get("images") or [])
                if isinstance(image, dict) and self._norm_bbox(image.get("bbox"))
            ],
        }

    def _build_inferred_structure(self, page_data, observed, columns):
        elements = list(observed.get("elements") or [])
        top_blocks = [el for el in elements if el.get("type") == "block"]
        line_elements = [el for el in elements if el.get("type") == "line"]
        top_blocks.sort(key=lambda el: (el.get("bbox", [0, 0, 0, 0])[1], el.get("bbox", [0, 0, 0, 0])[0], el.get("id")))

        sections = []
        section_memberships = []
        current_section_id = None
        for el in top_blocks:
            role = str(el.get("role") or "").strip().lower()
            if role in {"title", "section_heading"}:
                current_section_id = f"section::{el['id']}"
                sections.append(
                    {
                        "id": current_section_id,
                        "heading_id": el["id"],
                        "bbox": el.get("bbox"),
                        "column_index": el.get("column_index"),
                        "confidence": 0.96,
                        "evidence": ["heading_role"],
                    }
                )
            if current_section_id:
                section_memberships.append(
                    {
                        "section_id": current_section_id,
                        "member_id": el["id"],
                        "confidence": 0.92 if role in {"body", "paragraph", "list_item"} else 0.98,
                        "evidence": ["reading_order_proximity", "same_column"],
                    }
                )

        toc_entries = []
        toc_memberships = []
        if str(page_data.get("page_role") or "").strip().lower() == "toc":
            toc_rows = ((page_data.get("toc") or {}).get("toc_rows") or [])
            if toc_rows:
                for idx, row in enumerate(toc_rows):
                    if not isinstance(row, dict):
                        continue
                    role = str(row.get("role") or "").strip().lower()
                    if role in {"toc_title"}:
                        continue
                    row_bbox = self._toc_row_bbox(row)
                    if not row_bbox:
                        continue
                    entry_id = f"toc_entry::{idx}"
                    toc_entries.append(
                        {
                            "id": entry_id,
                            "bbox": row_bbox,
                            "confidence": 0.97,
                            "evidence": ["toc_rows", "toc_page_role"],
                            "role": role,
                            "page_value": str(row.get("page") or "").strip(),
                            "label": self._clean_text(row.get("translated_label") or row.get("label") or ""),
                        }
                    )
                    member_ids = self._toc_row_member_ids(row_bbox, top_blocks)
                    for member_id in member_ids:
                        toc_memberships.append(
                            {
                                "toc_entry_id": entry_id,
                                "member_id": member_id,
                                "confidence": 0.9,
                                "evidence": ["toc_rows_overlap"],
                            }
                        )
            else:
                row_buckets = {}
                for el in top_blocks:
                    bbox = el.get("bbox") or [0, 0, 0, 0]
                    key = int(round(((float(bbox[1]) + float(bbox[3])) * 0.5) / 18.0))
                    row_buckets.setdefault(key, []).append(el)
                for idx, members in enumerate(sorted(row_buckets.values(), key=lambda bucket: min(m["bbox"][1] for m in bucket))):
                    if len(members) < 2:
                        continue
                    members = sorted(members, key=lambda el: (el["bbox"][0], el["id"]))
                    entry_id = f"toc_entry::{idx}"
                    toc_entries.append(
                        {
                            "id": entry_id,
                            "bbox": self._union_bbox([m.get("bbox") for m in members]),
                            "confidence": 0.93,
                            "evidence": ["same_row_alignment", "toc_page_role"],
                        }
                    )
                    for member in members:
                        toc_memberships.append(
                            {
                                "toc_entry_id": entry_id,
                                "member_id": member["id"],
                                "confidence": 0.91,
                                "evidence": ["same_row_alignment"],
                            }
                        )

        key_value_pairs = []
        if self._looks_like_abbreviation_page(page_data, top_blocks):
            by_row = {}
            for el in top_blocks:
                bbox = el.get("bbox") or [0, 0, 0, 0]
                row_key = int(round(((float(bbox[1]) + float(bbox[3])) * 0.5) / 16.0))
                by_row.setdefault(row_key, []).append(el)
            pair_idx = 0
            for members in by_row.values():
                members = sorted(members, key=lambda el: (el["bbox"][0], el["id"]))
                if len(members) < 2:
                    continue
                left = members[0]
                right = members[1]
                if self._looks_like_abbreviation_key(left.get("text")):
                    key_value_pairs.append(
                        {
                            "id": f"kv::{pair_idx}",
                            "key_id": left["id"],
                            "value_id": right["id"],
                            "bbox": self._union_bbox([left.get("bbox"), right.get("bbox")]),
                            "confidence": 0.94,
                            "evidence": ["same_row", "left_right_pairing", "abbreviation_key_shape"],
                        }
                    )
                    pair_idx += 1

        paragraph_chains = []
        for idx in range(len(top_blocks) - 1):
            cur = top_blocks[idx]
            nxt = top_blocks[idx + 1]
            if str(cur.get("role") or "") != "body" or str(nxt.get("role") or "") != "body":
                continue
            if cur.get("column_index") != nxt.get("column_index"):
                continue
            cur_bbox = cur.get("bbox") or [0, 0, 0, 0]
            nxt_bbox = nxt.get("bbox") or [0, 0, 0, 0]
            gap = max(0.0, float(nxt_bbox[1]) - float(cur_bbox[3]))
            if gap > 40.0:
                continue
            if abs(float(cur_bbox[0]) - float(nxt_bbox[0])) > 28.0:
                continue
            paragraph_chains.append(
                {
                    "source_id": cur["id"],
                    "target_id": nxt["id"],
                    "confidence": 0.9,
                    "evidence": ["same_column", "vertical_continuity", "left_alignment"],
                }
            )
        if not paragraph_chains:
            paragraph_chains.extend(self._build_dense_body_line_paragraph_chains(page_data, line_elements))

        chapter_openings = []
        chapter_opening_memberships = []
        chapter_signal = float(
            (((page_data.get("page_case_v2") or {}).get("page_archetype_signals") or {}).get("chapter_opening", 0.0) or 0.0)
        )
        if chapter_signal >= 0.6:
            title_blocks = [el for el in top_blocks if str(el.get("role") or "").strip().lower() in {"title", "subtitle"}]
            title_blocks = [el for el in title_blocks if (el.get("bbox") or [0, 0, 0, 0])[1] <= 420.0]
            title_blocks.sort(key=lambda el: (el.get("bbox", [0, 0, 0, 0])[1], el.get("bbox", [0, 0, 0, 0])[0], el.get("id")))
            if title_blocks:
                title_bottom = max(float((el.get("bbox") or [0, 0, 0, 0])[3]) for el in title_blocks)
                followers = []
                for el in top_blocks:
                    role = str(el.get("role") or "").strip().lower()
                    if role not in {"section_heading", "body"}:
                        continue
                    bbox = el.get("bbox") or [0, 0, 0, 0]
                    if float(bbox[1]) < title_bottom:
                        continue
                    followers.append(el)
                followers.sort(key=lambda el: (el.get("bbox", [0, 0, 0, 0])[1], el.get("bbox", [0, 0, 0, 0])[0], el.get("id")))
                member_blocks = list(title_blocks)
                member_blocks.extend(followers[:2])
                opening_id = "chapter_opening::0"
                chapter_openings.append(
                    {
                        "id": opening_id,
                        "bbox": self._union_bbox([member.get("bbox") for member in member_blocks]),
                        "confidence": round(min(0.98, chapter_signal), 4),
                        "evidence": ["classifier_v2_chapter_opening", "title_stack", "opening_followers"],
                    }
                )
                for member in member_blocks:
                    chapter_opening_memberships.append(
                        {
                            "chapter_opening_id": opening_id,
                            "member_id": member["id"],
                            "confidence": round(min(0.96, chapter_signal), 4),
                            "evidence": ["opening_cluster_membership"],
                        }
                    )

        if toc_entries:
            sections = []
            section_memberships = []
        elif key_value_pairs:
            sections = []
            section_memberships = []

        return {
            "sections": sections,
            "section_memberships": section_memberships,
            "toc_entries": toc_entries,
            "toc_memberships": toc_memberships,
            "key_value_pairs": key_value_pairs,
            "paragraph_chains": paragraph_chains,
            "chapter_openings": chapter_openings,
            "chapter_opening_memberships": chapter_opening_memberships,
        }

    def _build_synthetic_structure(self, page_data, columns, page_w, page_h):
        margins = ((page_data.get("layout") or {}).get("margins") or {})
        return {
            "page_guides": {
                "margins": margins,
                "columns": [
                    {
                        "id": f"guide_col_{int(col.get('id', idx) or idx)}",
                        "bbox": [float(col.get("x0", 0.0) or 0.0), 0.0, float(col.get("x1", page_w) or page_w), page_h],
                    }
                    for idx, col in enumerate(columns)
                ],
            },
            "keep_out_zones": [
                {
                    "id": f"keepout_{idx}",
                    "bbox": self._norm_bbox(image.get("bbox")),
                    "kind": "image",
                }
                for idx, image in enumerate(page_data.get("images") or [])
                if isinstance(image, dict) and self._norm_bbox(image.get("bbox"))
            ] + [
                {
                    "id": f"keepout_non_text_{idx}",
                    "bbox": self._norm_bbox(bbox),
                    "kind": "non_text",
                }
                for idx, bbox in enumerate(page_data.get("non_text_zones") or [])
                if self._norm_bbox(bbox)
            ],
        }

    def _build_hierarchy(self, observed, inferred, synthetic):
        nodes = []
        edges = []
        nodes.append({"id": "page::root", "type": "page"})
        for region in observed.get("regions") or []:
            nodes.append({"id": region["id"], "type": "region"})
            edges.append({"type": "contains", "source": "page::root", "target": region["id"], "observed": True})
        for element in observed.get("elements") or []:
            nodes.append({"id": element["id"], "type": element.get("type")})
            parent = element.get("parent_id")
            if parent:
                edges.append({"type": "contains", "source": parent, "target": element["id"], "observed": True})
            else:
                edges.append({"type": "contains", "source": "page::root", "target": element["id"], "observed": True})
        for span in observed.get("spans") or []:
            nodes.append({"id": span["id"], "type": "span"})
            edges.append({"type": "contains", "source": span.get("parent_id"), "target": span["id"], "observed": True})
        for section in inferred.get("sections") or []:
            nodes.append({"id": section["id"], "type": "section"})
            edges.append({"type": "contains", "source": "page::root", "target": section["id"], "observed": False})
            if section.get("heading_id"):
                edges.append({"type": "heads", "source": section["heading_id"], "target": section["id"], "observed": False})
        for opening in inferred.get("chapter_openings") or []:
            nodes.append({"id": opening["id"], "type": "chapter_opening"})
            edges.append({"type": "contains", "source": "page::root", "target": opening["id"], "observed": False})
        return {"nodes": nodes, "edges": edges}

    def _build_dependency_graph(self, page_data, observed, inferred):
        edges = []
        for membership in inferred.get("section_memberships") or []:
            edges.append(
                {
                    "type": "belongs_to_section",
                    "source": membership.get("member_id"),
                    "target": membership.get("section_id"),
                    "confidence": membership.get("confidence"),
                    "evidence": membership.get("evidence"),
                    "origin": "inferred",
                }
            )
        for link in inferred.get("paragraph_chains") or []:
            edges.append(
                {
                    "type": "continues_paragraph",
                    "source": link.get("source_id"),
                    "target": link.get("target_id"),
                    "confidence": link.get("confidence"),
                    "evidence": link.get("evidence"),
                    "origin": "inferred",
                }
            )
        for kv in inferred.get("key_value_pairs") or []:
            edges.append(
                {
                    "type": "key_for_value",
                    "source": kv.get("key_id"),
                    "target": kv.get("value_id"),
                    "confidence": kv.get("confidence"),
                    "evidence": kv.get("evidence"),
                    "origin": "inferred",
                }
            )
        for toc in inferred.get("toc_memberships") or []:
            edges.append(
                {
                    "type": "member_of_toc_entry",
                    "source": toc.get("member_id"),
                    "target": toc.get("toc_entry_id"),
                    "confidence": toc.get("confidence"),
                    "evidence": toc.get("evidence"),
                    "origin": "inferred",
                }
            )
        for opening in inferred.get("chapter_opening_memberships") or []:
            edges.append(
                {
                    "type": "member_of_chapter_opening",
                    "source": opening.get("member_id"),
                    "target": opening.get("chapter_opening_id"),
                    "confidence": opening.get("confidence"),
                    "evidence": opening.get("evidence"),
                    "origin": "inferred",
                }
            )
        for image in observed.get("images") or []:
            image_bbox = image.get("bbox")
            if not image_bbox:
                continue
            best = None
            best_dist = None
            for element in observed.get("elements") or []:
                if element.get("type") != "block":
                    continue
                role = str(element.get("role") or "").strip().lower()
                if role not in {"figure_caption", "title", "diagram_label", "diagram_text_label"}:
                    continue
                dist = self._bbox_distance(element.get("bbox"), image_bbox)
                if best_dist is None or dist < best_dist:
                    best = element
                    best_dist = dist
            if best is not None and best_dist is not None and best_dist <= 240.0:
                rel_type = "caption_for" if str(best.get("role") or "") == "figure_caption" else "label_for"
                edges.append(
                    {
                        "type": rel_type,
                        "source": best.get("id"),
                        "target": image.get("id"),
                        "confidence": 0.86,
                        "evidence": ["proximity", "visual_attachment"],
                        "origin": "inferred",
                    }
                )
        return {"edges": edges}

    def _build_spatial_graph(self, page_data, observed, inferred, columns):
        top_blocks = [el for el in observed.get("elements") or [] if el.get("type") == "block"]
        top_blocks.sort(key=lambda el: (el.get("bbox", [0, 0, 0, 0])[1], el.get("bbox", [0, 0, 0, 0])[0], el.get("id")))
        edges = []
        row_clusters = {}
        baseline_clusters = {}
        column_members = {}
        for idx in range(len(top_blocks)):
            a = top_blocks[idx]
            a_bbox = a.get("bbox") or [0, 0, 0, 0]
            column_members.setdefault(a.get("column_index"), []).append(a)
            row_key = (a.get("column_index"), int(round(((float(a_bbox[1]) + float(a_bbox[3])) * 0.5) / 12.0)))
            row_clusters.setdefault(row_key, []).append(a["id"])
            baseline_key = (
                a.get("column_index"),
                self._style_key(a),
                int(round(float(a_bbox[3]) / 8.0)),
            )
            baseline_clusters.setdefault(baseline_key, []).append(a["id"])
            for jdx in range(idx + 1, len(top_blocks)):
                b = top_blocks[jdx]
                b_bbox = b.get("bbox") or [0, 0, 0, 0]
                vertical_gap = max(0.0, float(b_bbox[1]) - float(a_bbox[3]))
                if vertical_gap > 260.0:
                    break
                if abs(float(a_bbox[0]) - float(b_bbox[0])) <= 8.0 and vertical_gap <= 220.0:
                    edges.append(self._spatial_edge("aligned_left", a["id"], b["id"], 0.9, ["x0_distance"]))
                if abs(float(a_bbox[2]) - float(b_bbox[2])) <= 8.0 and vertical_gap <= 220.0:
                    edges.append(self._spatial_edge("aligned_right", a["id"], b["id"], 0.9, ["x1_distance"]))
                a_mid = (float(a_bbox[1]) + float(a_bbox[3])) * 0.5
                b_mid = (float(b_bbox[1]) + float(b_bbox[3])) * 0.5
                if abs(a_mid - b_mid) <= 10.0:
                    edges.append(self._spatial_edge("same_row", a["id"], b["id"], 0.88, ["midline_proximity"]))
                if abs(a_mid - b_mid) <= 6.0 and self._style_key(a) == self._style_key(b):
                    edges.append(self._spatial_edge("shares_baseline", a["id"], b["id"], 0.84, ["midline_proximity", "style_similarity"]))
                if abs(self._center_x(a_bbox) - self._center_x(b_bbox)) <= 12.0 and vertical_gap <= 140.0:
                    edges.append(self._spatial_edge("centered_with", a["id"], b["id"], 0.8, ["center_x_proximity"]))

        for members in column_members.values():
            members = sorted(members, key=lambda el: (el.get("bbox", [0, 0, 0, 0])[1], el.get("bbox", [0, 0, 0, 0])[0], el.get("id")))
            for idx in range(len(members) - 1):
                a = members[idx]
                b = members[idx + 1]
                edges.append(self._spatial_edge("same_column", a["id"], b["id"], 0.98, ["column_index", "adjacent_in_column"]))

        return {
            "edges": edges,
            "row_clusters": [
                {"id": f"row_cluster::{idx}", "member_ids": sorted(set(member_ids))}
                for idx, member_ids in enumerate(row_clusters.values())
                if len(set(member_ids)) >= 2
            ],
            "baseline_clusters": [
                {"id": f"baseline_cluster::{idx}", "member_ids": sorted(set(member_ids))}
                for idx, member_ids in enumerate(baseline_clusters.values())
                if len(set(member_ids)) >= 2
            ],
        }

    def _build_typographic_graph(self, observed, inferred):
        blocks = [el for el in observed.get("elements") or [] if el.get("type") == "block"]
        groups = []
        by_signature = {}
        for block in blocks:
            sig = self._style_key(block)
            if not sig:
                continue
            by_signature.setdefault(sig, []).append(block)
        for idx, (_, members) in enumerate(sorted(by_signature.items(), key=lambda item: item[0])):
            if len(members) < 2:
                continue
            groups.append(
                {
                    "id": f"typo_group::{idx}",
                    "member_ids": [member["id"] for member in members],
                    "dominant_font_size": members[0].get("style", {}).get("font_size_px", 0.0),
                    "dominant_font_weight": members[0].get("style", {}).get("font_weight", 400),
                    "confidence": 0.9,
                    "evidence": ["style_signature_match"],
                }
            )
        return {"groups": groups}

    def _build_render_model(self, page_data, observed, inferred, dependency_graph, spatial_graph, typographic_graph, structure_arbitration):
        render_units = []
        containers = []
        page_role = str(page_data.get("page_role") or "").strip().lower()
        line_elements = [el for el in observed.get("elements") or [] if el.get("type") == "line"]
        if page_role == "toc":
            for entry in inferred.get("toc_entries") or []:
                containers.append(
                    {
                        "id": entry["id"],
                        "kind": "toc_entry",
                        "bbox": entry.get("bbox"),
                        "member_ids": [
                            edge.get("source")
                            for edge in dependency_graph.get("edges") or []
                            if edge.get("type") == "member_of_toc_entry" and edge.get("target") == entry["id"]
                        ],
                        "reflow_policy": "entry_locked",
                    }
                )
        for pair in inferred.get("key_value_pairs") or []:
            containers.append(
                {
                    "id": pair["id"],
                    "kind": "key_value_pair",
                    "bbox": pair.get("bbox"),
                    "member_ids": [pair.get("key_id"), pair.get("value_id")],
                    "reflow_policy": "pair_locked",
                }
            )
        for section in inferred.get("sections") or []:
            member_ids = [
                edge.get("source")
                for edge in dependency_graph.get("edges") or []
                if edge.get("type") == "belongs_to_section" and edge.get("target") == section["id"]
            ]
            containers.append(
                {
                    "id": section["id"],
                    "kind": "section",
                    "bbox": section.get("bbox"),
                    "member_ids": sorted(set(member_ids)),
                    "reflow_policy": "section_flow",
                }
            )
        for opening in inferred.get("chapter_openings") or []:
            member_ids = [
                edge.get("source")
                for edge in dependency_graph.get("edges") or []
                if edge.get("type") == "member_of_chapter_opening" and edge.get("target") == opening["id"]
            ]
            containers.append(
                {
                    "id": opening["id"],
                    "kind": "chapter_opening",
                    "bbox": opening.get("bbox"),
                    "member_ids": sorted(set(member_ids)),
                    "reflow_policy": "opening_locked",
                }
            )
        containers.extend(self._build_line_paragraph_containers(page_data, observed, inferred))
        active_container_ids = set(structure_arbitration.get("active_container_ids") or [])
        secondary_container_ids = set(structure_arbitration.get("secondary_container_ids") or [])
        if structure_arbitration.get("primary_structure_family") == "dense_paragraph_flow":
            for container in containers:
                if str(container.get("kind") or "") == "paragraph_segment" and container.get("id"):
                    active_container_ids.add(str(container.get("id")))
        for container in containers:
            cid = str(container.get("id") or "")
            container["structure_family"] = self._container_structure_family(container)
            container["structure_priority"] = (
                "primary" if cid in active_container_ids else ("secondary" if cid in secondary_container_ids else "auxiliary")
            )
            container["active"] = cid in active_container_ids
        for block in [el for el in observed.get("elements") or [] if el.get("type") == "block"]:
            role = str(block.get("role") or "").strip().lower()
            unit_kind = "anchored_label" if role in {"diagram_label", "diagram_text_label", "figure_caption", "header", "footer"} else "text_block"
            if str(page_data.get("page_role") or "").strip().lower() == "toc":
                unit_kind = "toc_row_member"
            container_ids = [
                container["id"]
                for container in containers
                if block["id"] in (container.get("member_ids") or [])
            ]
            render_units.append(
                {
                    "id": f"render::{block['id']}",
                    "source_element_id": block["id"],
                    "kind": unit_kind,
                    "bbox": block.get("bbox"),
                    "column_index": block.get("column_index"),
                    "hierarchy_dependencies": [
                        edge["target"]
                        for edge in dependency_graph.get("edges") or []
                        if edge.get("source") == block["id"] and edge.get("type") in {"belongs_to_section", "member_of_toc_entry", "key_for_value"}
                    ],
                    "spatial_dependencies": [
                        edge["target"]
                        for edge in spatial_graph.get("edges") or []
                        if edge.get("source") == block["id"] and edge.get("type") in {"same_row", "same_column", "aligned_left", "aligned_right", "shares_baseline"}
                    ],
                    "container_ids": container_ids,
                    "structure_priority": self._unit_structure_priority(container_ids, active_container_ids, secondary_container_ids),
                    "reflow_policy": self._render_reflow_policy(page_data, block, dependency_graph),
                    "confidence": 0.9,
                }
            )
        for line in line_elements:
            line_id = str(line.get("id") or "")
            if not line_id:
                continue
            container_ids = [
                container["id"]
                for container in containers
                if line_id in (container.get("member_ids") or [])
            ]
            if not container_ids:
                continue
            render_units.append(
                {
                    "id": f"render::{line_id}",
                    "source_element_id": line_id,
                    "kind": "line_flow_member",
                    "bbox": line.get("bbox"),
                    "column_index": line.get("column_index"),
                    "hierarchy_dependencies": [
                        edge["target"]
                        for edge in dependency_graph.get("edges") or []
                        if edge.get("source") == line_id and edge.get("type") in {"continues_paragraph", "belongs_to_section"}
                    ],
                    "spatial_dependencies": [],
                    "container_ids": container_ids,
                    "structure_priority": self._unit_structure_priority(container_ids, active_container_ids, secondary_container_ids),
                    "reflow_policy": "line_chain_locked",
                    "confidence": 0.82,
                }
            )
        return {
            "render_units": render_units,
            "containers": containers,
            "primary_structure_family": structure_arbitration.get("primary_structure_family"),
            "active_container_ids": sorted(active_container_ids),
            "priority_order": [unit["id"] for unit in sorted(render_units, key=lambda item: (item["bbox"][1], item["bbox"][0]))],
            "dependency_count": len(dependency_graph.get("edges") or []),
            "spatial_dependency_count": len(spatial_graph.get("edges") or []),
            "typographic_group_count": len(typographic_graph.get("groups") or []),
        }

    def _build_reconstruction_contract(
        self,
        page_data,
        observed,
        inferred,
        synthetic,
        dependency_graph,
        spatial_graph,
        typographic_graph,
        render_model,
        page_case_v2,
        structure_arbitration,
    ):
        classifier_risks = list((page_case_v2.get("risk_flags") or [])) if isinstance(page_case_v2, dict) else []
        reading_modes = (page_case_v2.get("reading_modes") or {}) if isinstance(page_case_v2, dict) else {}
        layout_tendencies = (page_case_v2.get("layout_tendencies") or {}) if isinstance(page_case_v2, dict) else {}

        placement_constraints = []
        for unit in render_model.get("render_units") or []:
            bbox = unit.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            placement_constraints.append(
                {
                    "unit_id": unit["id"],
                    "source_element_id": unit.get("source_element_id"),
                    "column_index": unit.get("column_index"),
                    "bbox_lock": bbox,
                    "reflow_policy": unit.get("reflow_policy"),
                    "container_ids": list(unit.get("container_ids") or []),
                    "must_follow_dependencies": bool(unit.get("hierarchy_dependencies") or unit.get("spatial_dependencies")),
                }
            )

        execution_edges = []
        for edge in dependency_graph.get("edges") or []:
            execution_edges.append(
                {
                    "type": edge.get("type"),
                    "source": edge.get("source"),
                    "target": edge.get("target"),
                    "confidence": edge.get("confidence"),
                    "execution_priority": self._execution_priority_for_dependency(edge.get("type")),
                }
            )
        for edge in spatial_graph.get("edges") or []:
            if edge.get("type") not in {"same_row", "aligned_left", "aligned_right", "shares_baseline", "same_column"}:
                continue
            execution_edges.append(
                {
                    "type": edge.get("type"),
                    "source": edge.get("source"),
                    "target": edge.get("target"),
                    "confidence": edge.get("confidence"),
                    "execution_priority": self._execution_priority_for_dependency(edge.get("type")),
                }
            )

        return {
            "version": "reconstruction_contract.v1",
            "page_role": str(page_data.get("page_role") or "body").strip().lower(),
            "primary_structure_family": structure_arbitration.get("primary_structure_family"),
            "structure_arbitration": structure_arbitration,
            "reading_modes": reading_modes,
            "layout_tendencies": layout_tendencies,
            "risk_flags": classifier_risks,
            "containers": render_model.get("containers") or [],
            "render_units": render_model.get("render_units") or [],
            "priority_order": render_model.get("priority_order") or [],
            "execution_edges": execution_edges,
            "placement_constraints": placement_constraints,
            "keep_out_zones": list((synthetic.get("keep_out_zones") or [])),
            "row_clusters": list(spatial_graph.get("row_clusters") or []),
            "baseline_clusters": list(spatial_graph.get("baseline_clusters") or []),
            "typographic_groups": list(typographic_graph.get("groups") or []),
            "contract_stats": {
                "render_unit_count": len(render_model.get("render_units") or []),
                "container_count": len(render_model.get("containers") or []),
                "execution_edge_count": len(execution_edges),
                "placement_constraint_count": len(placement_constraints),
            },
        }

    def _build_dense_body_line_paragraph_chains(self, page_data, line_elements):
        page_role = str(page_data.get("page_role") or "").strip().lower()
        if page_role == "toc":
            return []
        layout_type = str(page_data.get("layout_type") or "").strip().lower()
        page_family = str(page_data.get("page_family") or "").strip().lower()
        if layout_type == "table_dominant" or page_family in {"table_diagram_example", "table_page"}:
            return []
        grouped = {}
        for line in line_elements or []:
            role = str(line.get("role") or "").strip().lower()
            if role not in {"body", "paragraph", "list_item"}:
                continue
            parent_id = str(line.get("parent_id") or "")
            if not parent_id:
                continue
            grouped.setdefault(parent_id, []).append(line)
        chains = []
        for members in grouped.values():
            members.sort(key=lambda el: (el.get("bbox", [0, 0, 0, 0])[1], el.get("bbox", [0, 0, 0, 0])[0], el.get("id")))
            if len(members) < 4:
                continue
            for idx in range(len(members) - 1):
                cur = members[idx]
                nxt = members[idx + 1]
                cur_bbox = cur.get("bbox") or [0, 0, 0, 0]
                nxt_bbox = nxt.get("bbox") or [0, 0, 0, 0]
                gap = max(0.0, float(nxt_bbox[1]) - float(cur_bbox[3]))
                if gap > 24.0:
                    continue
                chains.append(
                    {
                        "source_id": cur["id"],
                        "target_id": nxt["id"],
                        "confidence": 0.78,
                        "evidence": ["dense_body_line_cluster", "same_parent_block", "vertical_continuity"],
                    }
                )
        return chains

    def _build_line_paragraph_containers(self, page_data, observed, inferred):
        page_role = str(page_data.get("page_role") or "").strip().lower()
        if page_role == "toc":
            return []
        layout_type = str(page_data.get("layout_type") or "").strip().lower()
        page_family = str(page_data.get("page_family") or "").strip().lower()
        if layout_type == "table_dominant" or page_family in {"table_diagram_example", "table_page"}:
            return []
        line_elements = [el for el in observed.get("elements") or [] if el.get("type") == "line"]
        if not line_elements:
            return []
        continues = [edge for edge in (inferred.get("paragraph_chains") or []) if "::line::" in str(edge.get("source_id") or "")]
        if not continues:
            return []
        by_parent = {}
        for line in line_elements:
            parent_id = str(line.get("parent_id") or "")
            if not parent_id:
                continue
            by_parent.setdefault(parent_id, []).append(line)
        containers = []
        for parent_id, members in by_parent.items():
            members.sort(key=lambda el: (el.get("bbox", [0, 0, 0, 0])[1], el.get("bbox", [0, 0, 0, 0])[0], el.get("id")))
            if len(members) < 4:
                continue
            bbox = self._union_bbox([member.get("bbox") for member in members])
            if not bbox:
                continue
            containers.append(
                {
                    "id": f"paragraph_segment::{parent_id}",
                    "kind": "paragraph_segment",
                    "bbox": bbox,
                    "member_ids": [member["id"] for member in members],
                    "reflow_policy": "paragraph_line_flow",
                }
            )
        return containers

    def _determine_primary_structure_family(self, page_data, inferred, page_case_v2):
        page_role = str(page_data.get("page_role") or "").strip().lower()
        if page_role == "toc" or (inferred.get("toc_entries") or []):
            return "toc"
        if inferred.get("key_value_pairs") or []:
            return "glossary_pairs"
        if inferred.get("chapter_openings") or []:
            return "chapter_opening"
        if any("::line::" in str(edge.get("source_id") or "") for edge in (inferred.get("paragraph_chains") or [])):
            return "dense_paragraph_flow"
        if inferred.get("sections") or []:
            return "section_flow"
        return "freeform_blocks"

    def _build_structure_arbitration(self, primary_structure_family, inferred):
        all_container_ids = []
        for entry in inferred.get("toc_entries") or []:
            all_container_ids.append((entry.get("id"), "toc_entry"))
        for pair in inferred.get("key_value_pairs") or []:
            all_container_ids.append((pair.get("id"), "key_value_pair"))
        for opening in inferred.get("chapter_openings") or []:
            all_container_ids.append((opening.get("id"), "chapter_opening"))
        for section in inferred.get("sections") or []:
            all_container_ids.append((section.get("id"), "section"))

        active_kinds = {
            "toc": {"toc_entry"},
            "glossary_pairs": {"key_value_pair"},
            "chapter_opening": {"chapter_opening"},
            "dense_paragraph_flow": {"paragraph_segment"},
            "section_flow": {"section"},
        }.get(primary_structure_family, set())
        secondary_kinds = {
            "toc": {"section"},
            "glossary_pairs": {"section"},
            "chapter_opening": {"section"},
            "dense_paragraph_flow": {"section"},
            "section_flow": set(),
        }.get(primary_structure_family, set())

        active_container_ids = [cid for cid, kind in all_container_ids if cid and kind in active_kinds]
        secondary_container_ids = [cid for cid, kind in all_container_ids if cid and kind in secondary_kinds]
        suppressed_inferred_collections = []
        if primary_structure_family == "toc":
            suppressed_inferred_collections.append("sections")
        if primary_structure_family == "glossary_pairs":
            suppressed_inferred_collections.append("sections")

        return {
            "primary_structure_family": primary_structure_family,
            "active_container_kinds": sorted(active_kinds),
            "secondary_container_kinds": sorted(secondary_kinds),
            "active_container_ids": sorted(active_container_ids),
            "secondary_container_ids": sorted(secondary_container_ids),
            "suppressed_inferred_collections": suppressed_inferred_collections,
        }

    def _container_structure_family(self, container):
        kind = str((container or {}).get("kind") or "").strip().lower()
        return {
            "toc_entry": "toc",
            "key_value_pair": "glossary_pairs",
            "chapter_opening": "chapter_opening",
            "paragraph_segment": "dense_paragraph_flow",
            "section": "section_flow",
        }.get(kind, "auxiliary")

    def _unit_structure_priority(self, container_ids, active_container_ids, secondary_container_ids):
        ids = {str(cid) for cid in (container_ids or [])}
        if ids & set(active_container_ids):
            return "primary"
        if ids & set(secondary_container_ids):
            return "secondary"
        return "auxiliary"

    def _toc_row_bbox(self, row):
        label_bbox = self._norm_bbox(row.get("label_bbox"))
        page_bbox = self._norm_bbox(row.get("page_bbox"))
        if label_bbox and page_bbox:
            return self._union_bbox([label_bbox, page_bbox])
        return label_bbox or page_bbox

    def _toc_row_member_ids(self, row_bbox, blocks):
        if not row_bbox:
            return []
        member_ids = []
        row_rect = row_bbox
        for block in blocks or []:
            block_bbox = self._norm_bbox(block.get("bbox"))
            if not block_bbox:
                continue
            overlap = self._bbox_intersection_ratio(row_rect, block_bbox)
            if overlap <= 0.0:
                continue
            if overlap >= 0.08 or self._bbox_contains_midline(block_bbox, row_rect):
                member_ids.append(str(block.get("id")))
        return sorted(set(member_ids))

    def _bbox_intersection_ratio(self, a, b):
        a = self._norm_bbox(a)
        b = self._norm_bbox(b)
        if not a or not b:
            return 0.0
        ax0, ay0, ax1, ay1 = [float(v) for v in a]
        bx0, by0, bx1, by1 = [float(v) for v in b]
        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter = (ix1 - ix0) * (iy1 - iy0)
        a_area = max(1.0, (ax1 - ax0) * (ay1 - ay0))
        b_area = max(1.0, (bx1 - bx0) * (by1 - by0))
        return inter / min(a_area, b_area)

    def _bbox_contains_midline(self, outer, inner):
        outer = self._norm_bbox(outer)
        inner = self._norm_bbox(inner)
        if not outer or not inner:
            return False
        mid_y = (float(inner[1]) + float(inner[3])) * 0.5
        return float(outer[1]) <= mid_y <= float(outer[3])

    def _render_reflow_policy(self, page_data, block, dependency_graph):
        role = str(block.get("role") or "").strip().lower()
        page_role = str(page_data.get("page_role") or "").strip().lower()
        if page_role == "toc":
            return "toc_row_locked"
        if role in {"figure_caption", "diagram_label", "diagram_text_label", "header", "footer"}:
            return "anchored_locked"
        if any(edge.get("source") == block.get("id") and edge.get("type") == "key_for_value" for edge in dependency_graph.get("edges") or []):
            return "pair_locked"
        if role in {"body", "paragraph", "list_item"}:
            return "paragraph_reflow"
        return "line_preserve"

    def _execution_priority_for_dependency(self, dep_type):
        dep = str(dep_type or "").strip().lower()
        if dep in {"member_of_toc_entry", "key_for_value", "caption_for", "label_for", "member_of_chapter_opening"}:
            return "hard"
        if dep in {"belongs_to_section", "continues_paragraph", "same_row", "shares_baseline"}:
            return "strong"
        return "soft"

    def _looks_like_abbreviation_page(self, page_data, blocks):
        heading_texts = []
        for block in blocks:
            role = str(block.get("role") or "").strip().lower()
            if role not in {"title", "section_heading", "header"}:
                continue
            text = self._clean_text(block.get("text"))
            if text:
                heading_texts.append(text.lower())
        if any(re.search(r"\b(abbreviations?|acronyms?|nomenclature|glossary)\b", text) for text in heading_texts):
            return True
        return str(page_data.get("page_role") or "").strip().lower() in {"glossary", "abbreviations"}

    def _looks_like_abbreviation_key(self, text):
        s = self._clean_text(text)
        if not s:
            return False
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-/]*", s)
        if not tokens or len(tokens) > 4:
            return False
        lexical = re.findall(r"[A-Za-z][A-Za-z0-9.\-/]*", s)
        if not lexical:
            return False
        return all(token.upper() == token or re.search(r"[\d./-]", token) for token in lexical)

    def _spatial_edge(self, rel_type, source, target, confidence, evidence):
        return {
            "type": rel_type,
            "source": source,
            "target": target,
            "confidence": round(float(confidence or 0.0), 4),
            "evidence": list(evidence or []),
            "origin": "inferred",
        }

    def _style_key(self, element):
        style = (element or {}).get("style") or {}
        size = round(float(style.get("font_size_px", style.get("size", 0.0)) or 0.0), 1)
        weight = int(style.get("font_weight", 400) or 400)
        align = str(style.get("align") or "left").strip().lower()
        if size <= 0.0:
            return ""
        return f"{size}|{weight}|{align}"

    def _style_payload(self, node):
        style = (node or {}).get("style") or {}
        flags = style.get("flags") or {}
        size = float(style.get("font_size_px", style.get("size", 0.0)) or 0.0)
        return {
            "font_size_px": size,
            "font_weight": 700 if (flags.get("bold") or style.get("font_weight", 400) >= 600) else int(style.get("font_weight", 400) or 400),
            "italic": bool(flags.get("italic") or style.get("italic")),
            "align": str(style.get("align") or "left").strip().lower(),
            "font_name": str(style.get("font") or ""),
        }

    def _column_index_for_bbox(self, bbox, columns):
        norm = self._norm_bbox(bbox)
        if not norm:
            return 0
        if not columns:
            return 0
        cx = self._center_x(norm)
        best_idx = 0
        best_dist = None
        for idx, col in enumerate(columns):
            x0 = float(col.get("x0", 0.0) or 0.0)
            x1 = float(col.get("x1", x0) or x0)
            center = (x0 + x1) * 0.5
            dist = abs(cx - center)
            if best_dist is None or dist < best_dist:
                best_idx = int(col.get("id", idx) or idx)
                best_dist = dist
        return best_idx

    def _center_x(self, bbox):
        return (float(bbox[0]) + float(bbox[2])) * 0.5

    def _bbox_distance(self, a, b):
        if not self._norm_bbox(a) or not self._norm_bbox(b):
            return 1e9
        ax0, ay0, ax1, ay1 = [float(v) for v in a]
        bx0, by0, bx1, by1 = [float(v) for v in b]
        dx = max(0.0, max(ax0 - bx1, bx0 - ax1))
        dy = max(0.0, max(ay0 - by1, by0 - ay1))
        return math.hypot(dx, dy)

    def _union_bbox(self, bboxes):
        valid = [self._norm_bbox(b) for b in bboxes if self._norm_bbox(b)]
        if not valid:
            return None
        return [
            min(b[0] for b in valid),
            min(b[1] for b in valid),
            max(b[2] for b in valid),
            max(b[3] for b in valid),
        ]

    def _norm_bbox(self, bbox):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            return [float(v) for v in bbox]
        except Exception:
            return None

    def _clean_text(self, text):
        return re.sub(r"\s+", " ", str(text or "")).strip()
