import math
import re


class LayoutDescriptorBuilder:
    VERSION = "layout_descriptor.v2"

    REGION_TYPE_MAP = {
        "title": "title",
        "section_header": "section_header",
        "section_heading": "section_header",
        "text": "text",
        "body": "text",
        "table": "table",
        "picture": "picture",
        "image": "picture",
        "figure": "picture",
        "caption": "caption",
        "figure_caption": "caption",
        "header": "header",
        "footer": "footer",
        "list_item": "list_item",
        "footnote": "footnote",
        "formula": "formula",
        "equation_inline": "formula",
        "equation_block": "formula",
        "sidebar": "sidebar",
    }

    def build(self, page_data):
        if not isinstance(page_data, dict):
            return {}

        dims = page_data.get("dimensions") or {}
        width = float(dims.get("width", 0.0) or 0.0)
        height = float(dims.get("height", 0.0) or 0.0)
        document_type = str(page_data.get("document_type") or "mixed_unknown")
        layout_type = str(page_data.get("layout_type") or "mixed_blocks")
        style_profile = str(page_data.get("style_profile") or "minimalist")
        page_role = str(page_data.get("page_role") or "body_page")
        confidence = page_data.get("classification_confidence") or {}

        regions = self._build_regions(page_data, width, height)
        region_map = {r["id"]: r for r in regions}
        elements, groups = self._build_elements(page_data, regions)
        relations = self._build_relations(elements, groups, regions)
        constraints = self._build_constraints(page_data, elements, groups, region_map)
        self._enrich_elements_structure(page_data, elements, groups, regions, relations, constraints)
        reading_order = self._build_reading_order(elements)
        features = self._build_features(page_data, elements, groups, regions, width, height)
        ai_structure = self._build_ai_structure(page_data, regions, elements, relations)
        native_structure = self._build_native_structure(page_data)
        page_organization = self._build_page_organization(page_data, regions, elements, groups, relations, ai_structure, native_structure)
        reconstruction_plan = self._build_reconstruction_plan(page_data, regions, elements, groups, constraints, page_organization)
        visual_text_model = self._build_visual_text_model(page_data, elements, page_organization)

        return {
            "page_id": int(page_data.get("page", 1) or 1) - 1,
            "page_number": int(page_data.get("page", 1) or 1),
            "page_size": {"width": width, "height": height, "unit": "px"},
            "document_type": document_type,
            "layout_type": layout_type,
            "style_profile": style_profile,
            "page_role": page_role,
            "classification_confidence": {
                "document_type": float(confidence.get("document_type", confidence.get("document", 0.0)) or 0.0),
                "layout_type": float(confidence.get("layout_type", confidence.get("layout", 0.0)) or 0.0),
                "style_profile": float(confidence.get("style_profile", confidence.get("style", 0.0)) or 0.0),
            },
            "regions": regions,
            "elements": elements,
            "groups": groups,
            "relations": relations,
            "constraints": constraints,
            "features": features,
            "ai_structure": ai_structure,
            "native_structure": native_structure,
            "visual_text_model": visual_text_model,
            "page_organization": page_organization,
            "reconstruction_plan": reconstruction_plan,
            "reading_order": reading_order,
            "descriptor_version": self.VERSION,
        }

    def _build_regions(self, page_data, page_w, page_h):
        regions = []
        layout = page_data.get("layout") or {}
        columns = layout.get("columns") or []
        for col in columns:
            bbox = [
                float(col.get("x0", 0.0) or 0.0),
                0.0,
                float(col.get("x1", page_w) or page_w),
                page_h,
            ]
            regions.append(
                {
                    "id": f"region_col_{int(col.get('id', len(regions)))}",
                    "type": "column",
                    "source": "synthetic_layout",
                    "bbox": bbox,
                    "column_index": int(col.get("id", len(regions))),
                    "parent_region_id": None,
                    "reading_order": len(regions),
                    "coverage_ratio": self._area(bbox) / max(1.0, page_w * page_h),
                    "dominant_element_type": "text_block",
                }
            )

        regions.extend(self._synthesize_table_regions(page_data, columns, page_w, page_h))
        regions.extend(self._synthesize_annotated_regions(page_data, columns, page_w, page_h))
        regions.extend(self._synthesize_chart_regions(page_data, columns, page_w, page_h))

        raw_regions = list(page_data.get("regions") or []) + list(page_data.get("ai_layout_regions") or [])
        for idx, region in enumerate(raw_regions):
            bbox = self._norm_bbox(region.get("bbox"))
            if not bbox:
                continue
            r_type = str(region.get("type") or region.get("role") or "unknown").strip().lower()
            mapped = self.REGION_TYPE_MAP.get(r_type, r_type or "unknown")
            regions.append(
                {
                    "id": str(region.get("id") or f"region_{mapped}_{idx}"),
                    "type": mapped,
                    "source": str(region.get("source") or "native_layout"),
                    "bbox": bbox,
                    "column_index": self._column_index_for_bbox(bbox, columns),
                    "parent_region_id": None,
                    "reading_order": len(regions),
                    "coverage_ratio": self._area(bbox) / max(1.0, page_w * page_h),
                    "dominant_element_type": mapped,
                }
            )

        if not regions:
            regions.append(
                {
                    "id": "region_page_main",
                    "type": "text",
                    "source": "synthetic_layout",
                    "bbox": [0.0, 0.0, page_w, page_h],
                    "column_index": 0,
                    "parent_region_id": None,
                    "reading_order": 0,
                    "coverage_ratio": 1.0,
                    "dominant_element_type": "text_block",
                }
            )
        return regions

    def _synthesize_chart_regions(self, page_data, columns, page_w, page_h):
        chart = page_data.get("chart_structure") or {}
        chart_bbox = self._norm_bbox(chart.get("chart_area_bbox"))
        if not chart_bbox:
            return []

        regions = [
            {
                "id": "region_chart_area_0",
                "type": "chart_area",
                "source": "synthetic_layout",
                "bbox": chart_bbox,
                "column_index": self._column_index_for_bbox(chart_bbox, columns),
                "parent_region_id": None,
                "reading_order": 8_500,
                "coverage_ratio": self._area(chart_bbox) / max(1.0, page_w * page_h),
                "dominant_element_type": "chart",
            }
        ]
        plot_bbox = self._norm_bbox(chart.get("plot_area_bbox"))
        if plot_bbox:
            regions.append(
                {
                    "id": "region_chart_plot_area_0",
                    "type": "chart_plot_area",
                    "source": "synthetic_layout",
                    "bbox": plot_bbox,
                    "column_index": self._column_index_for_bbox(plot_bbox, columns),
                    "parent_region_id": "region_chart_area_0",
                    "reading_order": 8_505,
                    "coverage_ratio": self._area(plot_bbox) / max(1.0, page_w * page_h),
                    "dominant_element_type": "chart_plot",
                }
            )

        blocks_by_id = {str(b.get("id") or ""): b for b in page_data.get("blocks") or []}

        def union_bbox(ids):
            rect = None
            for bid in ids or []:
                block = blocks_by_id.get(str(bid))
                if not block:
                    continue
                bbox = self._norm_bbox(block.get("bbox"))
                if not bbox:
                    continue
                rect = bbox if rect is None else [
                    min(rect[0], bbox[0]),
                    min(rect[1], bbox[1]),
                    max(rect[2], bbox[2]),
                    max(rect[3], bbox[3]),
                ]
            return rect

        y_ticks_bbox = union_bbox(chart.get("y_tick_block_ids"))
        if y_ticks_bbox:
            regions.append(
                {
                    "id": "region_chart_y_ticks_0",
                    "type": "chart_y_ticks",
                    "source": "synthetic_layout",
                    "bbox": y_ticks_bbox,
                    "column_index": self._column_index_for_bbox(y_ticks_bbox, columns),
                    "parent_region_id": "region_chart_area_0",
                    "reading_order": 8_510,
                    "coverage_ratio": self._area(y_ticks_bbox) / max(1.0, page_w * page_h),
                    "dominant_element_type": "tick_label",
                }
            )

        y_axis_bbox = union_bbox(chart.get("y_axis_label_ids"))
        if y_axis_bbox:
            regions.append(
                {
                    "id": "region_chart_y_axis_0",
                    "type": "chart_y_axis",
                    "source": "synthetic_layout",
                    "bbox": y_axis_bbox,
                    "column_index": self._column_index_for_bbox(y_axis_bbox, columns),
                    "parent_region_id": "region_chart_area_0",
                    "reading_order": 8_520,
                    "coverage_ratio": self._area(y_axis_bbox) / max(1.0, page_w * page_h),
                    "dominant_element_type": "axis_label",
                }
            )

        x_axis_bbox = union_bbox(chart.get("x_axis_label_ids"))
        if x_axis_bbox:
            regions.append(
                {
                    "id": "region_chart_x_axis_0",
                    "type": "chart_x_axis",
                    "source": "synthetic_layout",
                    "bbox": x_axis_bbox,
                    "column_index": self._column_index_for_bbox(x_axis_bbox, columns),
                    "parent_region_id": "region_chart_area_0",
                    "reading_order": 8_530,
                    "coverage_ratio": self._area(x_axis_bbox) / max(1.0, page_w * page_h),
                    "dominant_element_type": "axis_label",
                }
            )

        x_ticks_bbox = union_bbox(chart.get("x_tick_block_ids"))
        if x_ticks_bbox:
            regions.append(
                {
                    "id": "region_chart_x_ticks_0",
                    "type": "chart_x_ticks",
                    "source": "synthetic_layout",
                    "bbox": x_ticks_bbox,
                    "column_index": self._column_index_for_bbox(x_ticks_bbox, columns),
                    "parent_region_id": "region_chart_area_0",
                    "reading_order": 8_535,
                    "coverage_ratio": self._area(x_ticks_bbox) / max(1.0, page_w * page_h),
                    "dominant_element_type": "tick_label",
                }
            )

        legend_bbox = union_bbox(chart.get("legend_label_ids"))
        if legend_bbox:
            regions.append(
                {
                    "id": "region_chart_legend_0",
                    "type": "chart_legend",
                    "source": "synthetic_layout",
                    "bbox": legend_bbox,
                    "column_index": self._column_index_for_bbox(legend_bbox, columns),
                    "parent_region_id": "region_chart_area_0",
                    "reading_order": 8_540,
                    "coverage_ratio": self._area(legend_bbox) / max(1.0, page_w * page_h),
                    "dominant_element_type": "legend_label",
                }
            )

        return regions

    def _synthesize_annotated_regions(self, page_data, columns, page_w, page_h):
        layout_type = str(page_data.get("layout_type") or "").strip().lower()
        style_profile = str(page_data.get("style_profile") or "").strip().lower()
        if layout_type != "annotated_page" and style_profile != "editorial_visual":
            return []

        rects = []
        for bbox in page_data.get("non_text_zones") or []:
            nb = self._norm_bbox(bbox)
            if nb and self._area(nb) >= max(1200.0, page_w * page_h * 0.003):
                rects.append(nb)
        for image in page_data.get("images") or []:
            if not isinstance(image, dict):
                continue
            nb = self._norm_bbox(image.get("bbox"))
            if nb and self._area(nb) >= max(1200.0, page_w * page_h * 0.003):
                rects.append(nb)
        if not rects:
            return []

        clusters = []
        for bbox in sorted(rects, key=lambda b: (b[1], b[0], self._area(b))):
            placed = False
            for cluster in clusters:
                cb = cluster["bbox"]
                overlap = self._intersection_area(bbox, cb)
                x_gap = max(0.0, max(cb[0] - bbox[2], bbox[0] - cb[2]))
                y_gap = max(0.0, max(cb[1] - bbox[3], bbox[1] - cb[3]))
                if overlap > 0 or (x_gap <= 40.0 and y_gap <= 40.0):
                    cluster["bbox"] = [
                        min(cb[0], bbox[0]),
                        min(cb[1], bbox[1]),
                        max(cb[2], bbox[2]),
                        max(cb[3], bbox[3]),
                    ]
                    cluster["count"] += 1
                    placed = True
                    break
            if not placed:
                clusters.append({"bbox": list(bbox), "count": 1})

        regions = []
        kept_clusters = []
        for idx, cluster in enumerate(clusters):
            bbox = cluster["bbox"]
            area = self._area(bbox)
            if area < max(2400.0, page_w * page_h * 0.01):
                continue
            kept_clusters.append({"id": f"region_illustration_{idx}", "bbox": bbox})
            regions.append(
                {
                    "id": f"region_illustration_{idx}",
                    "type": "illustration",
                    "source": "synthetic_layout",
                    "bbox": bbox,
                    "column_index": self._column_index_for_bbox(bbox, columns),
                    "parent_region_id": None,
                    "reading_order": 9_000 + idx,
                    "coverage_ratio": area / max(1.0, page_w * page_h),
                    "dominant_element_type": "figure",
                }
            )

        blocks = page_data.get("blocks") or []
        text_band_idx = 0
        annotation_idx = 0
        caption_idx = 0
        header_idx = 0
        for block in blocks:
            bbox = self._norm_bbox(block.get("bbox"))
            if not bbox:
                continue
            role = str(block.get("role") or "").strip().lower()
            unit_type = str(block.get("unit_type") or self._default_unit_type(role)).strip().lower()
            text = self._clean_text(block.get("translated_text") or block.get("text") or "")
            words = len(re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", text))
            nearest = None
            nearest_dist = None
            for cluster in kept_clusters:
                dist = self._bbox_distance(bbox, cluster["bbox"])
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
                    nearest = cluster
            if role in {"header", "title", "section_heading"} and bbox[1] <= page_h * 0.18:
                header_band = [
                    max(0.0, bbox[0] - 14.0),
                    max(0.0, bbox[1] - 8.0),
                    min(page_w, bbox[2] + 14.0),
                    min(page_h, bbox[3] + 10.0),
                ]
                regions.append(
                    {
                        "id": f"region_header_band_{header_idx}",
                        "type": "header_band",
                        "source": "synthetic_layout",
                        "bbox": header_band,
                        "column_index": self._column_index_for_bbox(header_band, columns),
                        "parent_region_id": None,
                        "reading_order": 9_100 + header_idx,
                        "coverage_ratio": self._area(header_band) / max(1.0, page_w * page_h),
                        "dominant_element_type": "header",
                    }
                )
                header_idx += 1
                continue
            if role == "figure_caption":
                text_band = [
                    max(0.0, bbox[0] - 12.0),
                    max(0.0, bbox[1] - 8.0),
                    min(page_w, bbox[2] + 12.0),
                    min(page_h, bbox[3] + 10.0),
                ]
                parent_id = nearest["id"] if nearest is not None and nearest_dist is not None and nearest_dist <= 120.0 else None
                regions.append(
                    {
                        "id": f"region_caption_band_{caption_idx}",
                        "type": "caption_band",
                        "source": "synthetic_layout",
                        "bbox": text_band,
                        "column_index": self._column_index_for_bbox(text_band, columns),
                        "parent_region_id": parent_id,
                        "reading_order": 9_500 + caption_idx,
                        "coverage_ratio": self._area(text_band) / max(1.0, page_w * page_h),
                        "dominant_element_type": "caption",
                    }
                )
                caption_idx += 1
                continue
            if role == "body" and words >= 12:
                band_bottom = bbox[3] + max(42.0, (bbox[3] - bbox[1]) * 0.7)
                if nearest is not None:
                    nearest_bbox = nearest.get("bbox") or [0, 0, 0, 0]
                    same_lane = self._intersection_area(
                        [bbox[0], 0.0, bbox[2], page_h],
                        [nearest_bbox[0], 0.0, nearest_bbox[2], page_h],
                    ) > 0
                    if same_lane and nearest_bbox[1] > bbox[1]:
                        band_bottom = min(band_bottom, nearest_bbox[1] - 12.0)
                text_band = [
                    max(0.0, bbox[0] - 18.0),
                    max(0.0, bbox[1] - 8.0),
                    min(page_w, bbox[2] + 18.0),
                    min(page_h, max(bbox[3] + 18.0, band_bottom)),
                ]
                regions.append(
                    {
                        "id": f"region_text_band_{text_band_idx}",
                        "type": "text_band",
                        "source": "synthetic_layout",
                        "bbox": text_band,
                        "column_index": self._column_index_for_bbox(text_band, columns),
                        "parent_region_id": None,
                        "reading_order": 9_200 + text_band_idx,
                        "coverage_ratio": self._area(text_band) / max(1.0, page_w * page_h),
                        "dominant_element_type": "text_block",
                    }
                )
                text_band_idx += 1
                continue
            if role in {"title", "diagram_label", "diagram_text_label", "section_heading"} or unit_type in {"short_label", "chart_label", "diagram_label"}:
                if nearest is not None and nearest_dist is not None and nearest_dist <= 220.0:
                    band_bbox = [
                        max(0.0, bbox[0] - 10.0),
                        max(0.0, bbox[1] - 6.0),
                        min(page_w, bbox[2] + 10.0),
                        min(page_h, bbox[3] + 6.0),
                    ]
                    regions.append(
                        {
                            "id": f"region_annotation_band_{annotation_idx}",
                            "type": "annotation_band",
                            "source": "synthetic_layout",
                            "bbox": band_bbox,
                            "column_index": self._column_index_for_bbox(band_bbox, columns),
                            "parent_region_id": nearest["id"],
                            "reading_order": 9_300 + annotation_idx,
                            "coverage_ratio": self._area(band_bbox) / max(1.0, page_w * page_h),
                            "dominant_element_type": "annotation",
                        }
                    )
                    annotation_idx += 1
        return regions

    def _synthesize_table_regions(self, page_data, columns, page_w, page_h):
        layout_type = str(page_data.get("layout_type") or "").strip().lower()
        document_type = str(page_data.get("document_type") or "").strip().lower()
        if layout_type != "table_dominant" and document_type not in {"form", "invoice", "receipt"}:
            return []

        candidate_blocks = []
        for b_idx, block in enumerate(page_data.get("blocks") or []):
            bbox = self._norm_bbox(block.get("bbox"))
            if not bbox:
                continue
            role = str(block.get("role") or "body").strip().lower()
            if role in {"footer", "page_number"}:
                continue
            candidate_blocks.append(
                {
                    "id": str(block.get("id") or f"blk_{b_idx}"),
                    "bbox": bbox,
                    "role": role,
                }
            )
        if len(candidate_blocks) < 3:
            return []

        candidate_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
        rows = []
        for block in candidate_blocks:
            placed = False
            by0, by1 = block["bbox"][1], block["bbox"][3]
            bmid = (by0 + by1) / 2.0
            for row in rows:
                ry0, ry1 = row["bbox"][1], row["bbox"][3]
                rmid = (ry0 + ry1) / 2.0
                tol = max(18.0, min((by1 - by0), (ry1 - ry0)) * 0.9)
                if abs(bmid - rmid) <= tol:
                    row["blocks"].append(block)
                    row["bbox"] = [
                        min(row["bbox"][0], block["bbox"][0]),
                        min(row["bbox"][1], block["bbox"][1]),
                        max(row["bbox"][2], block["bbox"][2]),
                        max(row["bbox"][3], block["bbox"][3]),
                    ]
                    placed = True
                    break
            if not placed:
                rows.append({"blocks": [block], "bbox": list(block["bbox"])})

        if len(rows) < 2:
            return []

        table_bbox = [
            min(r["bbox"][0] for r in rows),
            min(r["bbox"][1] for r in rows),
            max(r["bbox"][2] for r in rows),
            max(r["bbox"][3] for r in rows),
        ]
        regions = [
            {
                "id": "region_table_main",
                "type": "table",
                "source": "synthetic_layout",
                "bbox": table_bbox,
                "column_index": self._column_index_for_bbox(table_bbox, columns),
                "parent_region_id": None,
                "reading_order": 10_000,
                "coverage_ratio": self._area(table_bbox) / max(1.0, page_w * page_h),
                "dominant_element_type": "table",
            }
        ]
        for row_idx, row in enumerate(rows):
            row_bbox = row["bbox"]
            row_id = f"region_table_row_{row_idx}"
            regions.append(
                {
                    "id": row_id,
                    "type": "table_row",
                    "source": "synthetic_layout",
                    "bbox": row_bbox,
                    "column_index": self._column_index_for_bbox(row_bbox, columns),
                    "parent_region_id": "region_table_main",
                    "reading_order": 10_001 + row_idx,
                    "coverage_ratio": self._area(row_bbox) / max(1.0, page_w * page_h),
                    "dominant_element_type": "table_row",
                }
            )
            for cell_idx, block in enumerate(sorted(row["blocks"], key=lambda b: (b["bbox"][0], b["bbox"][1]))):
                cell_bbox = block["bbox"]
                regions.append(
                    {
                        "id": f"region_table_cell_{block['id']}",
                        "type": "table_cell",
                        "source": "synthetic_layout",
                        "bbox": cell_bbox,
                        "column_index": self._column_index_for_bbox(cell_bbox, columns),
                        "parent_region_id": row_id,
                        "reading_order": 11_000 + row_idx * 100 + cell_idx,
                        "coverage_ratio": self._area(cell_bbox) / max(1.0, page_w * page_h),
                        "dominant_element_type": "table_cell",
                    }
                )
        return regions

    def _build_elements(self, page_data, regions):
        elements = []
        groups = []
        columns = (page_data.get("layout") or {}).get("columns") or []
        blocks = page_data.get("blocks") or []
        sentence_counter = 0
        paragraph_counter = 0

        for b_idx, block in enumerate(blocks):
            bbox = self._norm_bbox(block.get("bbox"))
            if not bbox:
                continue
            block_id = str(block.get("id") or f"blk_{b_idx}")
            block_role = str(block.get("role") or "body").strip().lower()
            block_type = self._element_type_for_role(block_role)
            page_region_id = self._assign_region_id(page_data, block, bbox, regions)
            ai_region_id = self._best_ai_region_id(bbox, regions)
            column_index = self._column_index_for_bbox(bbox, columns)
            block_text = self._clean_text(block.get("translated_text") or block.get("text") or "")
            paragraph_id = None
            if block_role in {"body", "paragraph", "list_item"}:
                paragraph_counter += 1
                paragraph_id = f"para_{paragraph_counter}"

            line_ids = []
            block_element = {
                "id": block_id,
                "type": block_type,
                "role": block_role,
                "source": str(block.get("source") or "ocr"),
                "bbox": bbox,
                "polygon": None,
                "baseline": None,
                "center": self._center(bbox),
                "z_index": 3,
                "page_region_id": page_region_id,
                "ai_region_id": ai_region_id,
                "column_index": column_index,
                "parent_id": None,
                "children_ids": line_ids,
                "reading_order": 0,
                "paragraph_id": paragraph_id,
                "sentence_id": None,
                "sentence_index_in_paragraph": None,
                "line_index_in_block": None,
                "style": self._style_payload(block.get("style") or {}),
                "text": self._text_payload(block, block_text),
                "semantic": self._semantic_payload(block, block_role),
                "structure_hints": dict(block.get("structure_hints") or {}),
            }
            elements.append(block_element)

            sentence_index_in_paragraph = 0
            for l_idx, line in enumerate(block.get("lines") or []):
                line_bbox = self._norm_bbox(line.get("bbox")) or bbox
                line_id = f"{block_id}_ln_{l_idx}"
                line_ids.append(line_id)
                line_text = self._clean_text(line.get("translated_text") or line.get("line_text") or "")
                phrase_ids = []
                line_element = {
                    "id": line_id,
                    "type": "text_line",
                    "role": block_role,
                    "source": str(block.get("source") or "ocr"),
                    "bbox": line_bbox,
                    "polygon": None,
                    "baseline": [line_bbox[0], line_bbox[3], line_bbox[2], line_bbox[3]],
                    "center": self._center(line_bbox),
                    "z_index": 4,
                    "page_region_id": page_region_id,
                    "ai_region_id": ai_region_id,
                    "column_index": column_index,
                    "parent_id": block_id,
                    "children_ids": phrase_ids,
                    "reading_order": 0,
                    "paragraph_id": paragraph_id,
                    "sentence_id": None,
                    "sentence_index_in_paragraph": None,
                    "line_index_in_block": l_idx,
                    "style": self._style_payload(self._line_style(block, line)),
                    "text": self._text_payload(line, line_text),
                    "semantic": self._semantic_payload(block, block_role),
                    "structure_hints": dict(block.get("structure_hints") or {}),
                }
                elements.append(line_element)

                for p_idx, phrase in enumerate(line.get("phrases") or []):
                    phrase_bbox = self._norm_bbox(phrase.get("bbox")) or line_bbox
                    phrase_id = f"{line_id}_ph_{p_idx}"
                    phrase_ids.append(phrase_id)
                    sentence_counter += 1
                    sentence_index_in_paragraph += 1
                    phrase_text = self._clean_text(
                        phrase.get("translated_text")
                        or phrase.get("text")
                        or phrase.get("texte")
                        or line_text
                    )
                    span_ids = []
                    phrase_element = {
                        "id": phrase_id,
                        "type": "text_phrase",
                        "role": block_role,
                        "source": str(block.get("source") or "ocr"),
                        "bbox": phrase_bbox,
                        "polygon": None,
                        "baseline": [phrase_bbox[0], phrase_bbox[3], phrase_bbox[2], phrase_bbox[3]],
                        "center": self._center(phrase_bbox),
                        "z_index": 5,
                        "page_region_id": page_region_id,
                        "ai_region_id": ai_region_id,
                        "column_index": column_index,
                        "parent_id": line_id,
                        "children_ids": span_ids,
                        "reading_order": 0,
                        "paragraph_id": paragraph_id,
                        "sentence_id": f"sent_{sentence_counter}",
                        "sentence_index_in_paragraph": sentence_index_in_paragraph,
                        "line_index_in_block": l_idx,
                        "style": self._style_payload(self._phrase_style(block, phrase)),
                        "text": self._text_payload(phrase, phrase_text),
                        "semantic": self._semantic_payload(phrase, block_role, block),
                        "structure_hints": dict(block.get("structure_hints") or {}),
                    }
                    elements.append(phrase_element)

                    for s_idx, span in enumerate(phrase.get("spans") or []):
                        span_bbox = self._norm_bbox(span.get("bbox")) or phrase_bbox
                        span_id = f"{phrase_id}_sp_{s_idx}"
                        span_ids.append(span_id)
                        span_text = self._clean_text(
                            span.get("translated_text") or span.get("text") or span.get("texte") or ""
                        )
                        elements.append(
                            {
                                "id": span_id,
                                "type": "text_span",
                                "role": block_role,
                                "source": str(block.get("source") or "ocr"),
                                "bbox": span_bbox,
                                "polygon": None,
                                "baseline": [span_bbox[0], span_bbox[3], span_bbox[2], span_bbox[3]],
                                "center": self._center(span_bbox),
                                "z_index": 6,
                                "page_region_id": page_region_id,
                                "ai_region_id": ai_region_id,
                                "column_index": column_index,
                                "parent_id": phrase_id,
                                "children_ids": [],
                                "reading_order": 0,
                                "paragraph_id": paragraph_id,
                                "sentence_id": phrase_element["sentence_id"],
                                "sentence_index_in_paragraph": sentence_index_in_paragraph,
                                "line_index_in_block": l_idx,
                                "style": self._style_payload(span.get("style") or phrase.get("style") or {}),
                                "text": self._text_payload(span, span_text),
                                "semantic": self._semantic_payload(span, block_role, block),
                                "structure_hints": dict(block.get("structure_hints") or {}),
                            }
                        )

            if paragraph_id:
                child_phrase_ids = [
                    el["id"]
                    for el in elements
                    if el.get("paragraph_id") == paragraph_id and el.get("type") == "text_phrase"
                ]
                groups.append(
                    {
                        "id": paragraph_id,
                        "type": "paragraph",
                        "element_ids": child_phrase_ids,
                        "region_id": page_region_id,
                        "ai_region_id": ai_region_id,
                        "column_index": column_index,
                        "sentence_ids": [
                            el["sentence_id"]
                            for el in elements
                            if el.get("paragraph_id") == paragraph_id and el.get("sentence_id")
                        ],
                        "constraints": {
                            "render_mode": "flow_in_region",
                            "can_break_inside_sentence": False,
                            "allow_vertical_expand": bool(
                                block_role == "body" and len(re.findall(r"[A-Za-zÀ-ÿ]", block_text or "")) >= 40
                            ),
                        },
                    }
                )

        for i_idx, image in enumerate(page_data.get("images") or []):
            bbox = self._norm_bbox(image.get("bbox"))
            if not bbox:
                continue
            elements.append(
                {
                    "id": str(image.get("id") or f"fig_{i_idx}"),
                    "type": "figure",
                    "role": "figure",
                    "source": "native",
                    "bbox": bbox,
                    "polygon": None,
                    "baseline": None,
                    "center": self._center(bbox),
                    "z_index": 2,
                    "page_region_id": self._best_region_id(bbox, regions),
                    "ai_region_id": self._best_ai_region_id(bbox, regions),
                    "column_index": self._column_index_for_bbox(bbox, columns),
                    "parent_id": None,
                    "children_ids": [],
                    "reading_order": 0,
                    "paragraph_id": None,
                    "sentence_id": None,
                    "sentence_index_in_paragraph": None,
                    "line_index_in_block": None,
                    "style": {},
                    "text": {
                        "source_text": "",
                        "visible_text": "",
                        "translated_text": None,
                        "language": "",
                        "tokens": 0,
                        "is_truncated_source": False,
                    },
                    "semantic": {
                        "unit_type": "figure",
                        "is_translatable": False,
                        "is_reference_like": False,
                        "is_code_like": False,
                        "is_formula_like": False,
                    },
                    "structure_hints": {},
                }
            )

        reading_sorted = sorted(
            enumerate(elements),
            key=lambda kv: (
                float(kv[1].get("bbox", [0, 0, 0, 0])[1]),
                float(kv[1].get("bbox", [0, 0, 0, 0])[0]),
                kv[0],
            ),
        )
        for order, (_, element) in enumerate(reading_sorted):
            element["reading_order"] = order

        return elements, groups

    def _build_relations(self, elements, groups, regions):
        relations = []
        elements_by_id = {el["id"]: el for el in elements}
        region_map = {r["id"]: r for r in regions}
        top_level = [el for el in elements if not el.get("parent_id") and el.get("type") != "text_span"]
        top_level.sort(key=lambda el: (el.get("reading_order", 0), el["id"]))

        for element in elements:
            parent_id = element.get("parent_id")
            if parent_id and parent_id in elements_by_id:
                relations.append(self._rel("inside", element["id"], parent_id, weight=1.0))

        for idx in range(len(top_level) - 1):
            relations.append(self._rel("follows_in_reading_order", top_level[idx]["id"], top_level[idx + 1]["id"], weight=1.0))

        by_column = {}
        for element in top_level:
            col = element.get("column_index")
            by_column.setdefault(col, []).append(element)
        for col_elements in by_column.values():
            col_elements.sort(key=lambda el: (el["bbox"][1], el["bbox"][0]))
            for idx in range(len(col_elements) - 1):
                relations.append(self._rel("below", col_elements[idx]["id"], col_elements[idx + 1]["id"], weight=0.8))
                relations.append(self._rel("same_column", col_elements[idx]["id"], col_elements[idx + 1]["id"], weight=1.0))

        figures = [el for el in top_level if el.get("type") == "figure"]
        captions = [el for el in top_level if el.get("role") == "figure_caption" or el.get("type") == "caption"]
        for caption in captions:
            best = None
            best_score = None
            cb = caption["bbox"]
            for figure in figures:
                fb = figure["bbox"]
                dy = abs(cb[1] - fb[3])
                dx = abs(self._center(cb)[0] - self._center(fb)[0])
                score = dy + dx * 0.2
                if best_score is None or score < best_score:
                    best_score = score
                    best = figure
            if best is not None:
                relations.append(self._rel("caption_of", caption["id"], best["id"], weight=0.9))
                relations.append(self._rel("has_caption", best["id"], caption["id"], weight=0.9))

        for element in top_level:
            ai_region_id = element.get("ai_region_id")
            if ai_region_id and ai_region_id in region_map:
                relations.append(self._rel("inside_ai_region", element["id"], ai_region_id, weight=0.96))

        by_ai_region = {}
        for element in top_level:
            ai_region_id = element.get("ai_region_id")
            if ai_region_id:
                by_ai_region.setdefault(ai_region_id, []).append(element)
        for ai_elements in by_ai_region.values():
            if len(ai_elements) < 2:
                continue
            ai_elements.sort(key=lambda el: (el["bbox"][1], el["bbox"][0], el["id"]))
            for idx in range(len(ai_elements) - 1):
                relations.append(
                    self._rel("same_structural_band", ai_elements[idx]["id"], ai_elements[idx + 1]["id"], weight=0.9)
                )

        ai_title_regions = [
            r for r in regions
            if self._is_ai_region(r) and str(r.get("type") or "").strip().lower() in {"paragraph_title", "title", "header"}
        ]
        body_blocks = [
            el for el in top_level
            if el.get("type") == "text_block" and el.get("role") in {"body", "paragraph", "list_item"}
        ]
        for title_region in ai_title_regions:
            title_elements = [
                el for el in top_level
                if el.get("ai_region_id") == title_region.get("id")
                and el.get("role") in {"title", "section_heading", "header"}
            ]
            if not title_elements:
                continue
            heading = min(title_elements, key=lambda el: (el["bbox"][1], el["bbox"][0], el["id"]))
            heading_bbox = heading.get("bbox") or [0, 0, 0, 0]
            heading_center_x = self._center(heading_bbox)[0]
            candidate = None
            candidate_score = None
            for body in body_blocks:
                body_bbox = body.get("bbox") or [0, 0, 0, 0]
                if body_bbox[1] < heading_bbox[1]:
                    continue
                same_column = body.get("column_index") == heading.get("column_index")
                if not same_column and abs(self._center(body_bbox)[0] - heading_center_x) > 140.0:
                    continue
                vertical_gap = max(0.0, body_bbox[1] - heading_bbox[3])
                horizontal_gap = abs(self._center(body_bbox)[0] - heading_center_x)
                score = vertical_gap + horizontal_gap * 0.15
                if candidate_score is None or score < candidate_score:
                    candidate = body
                    candidate_score = score
            if candidate is not None and candidate_score is not None and candidate_score <= 220.0:
                relations.append(self._rel("title_of_region", heading["id"], title_region["id"], weight=0.94))
                relations.append(self._rel("heads_content", heading["id"], candidate["id"], weight=0.92))

        illustrations = [r for r in regions if str(r.get("type") or "") == "illustration"]
        anchored_labels = [
            el for el in top_level
            if el.get("role") in {"title", "diagram_text_label", "diagram_label"}
            or str((el.get("semantic") or {}).get("unit_type") or "") in {"short_label", "chart_label", "diagram_label"}
        ]
        for label in anchored_labels:
            lb = label.get("bbox") or [0, 0, 0, 0]
            best = None
            best_dist = None
            for region in illustrations:
                rb = region.get("bbox") or [0, 0, 0, 0]
                dist = self._bbox_distance(lb, rb)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best = region
            if best is not None and best_dist is not None and best_dist <= 220.0:
                relations.append(self._rel("anchored_to", label["id"], best["id"], weight=0.88))
                lc = self._center(lb)
                rc = self._center(best.get("bbox") or [0, 0, 0, 0])
                dx = lc[0] - rc[0]
                dy = lc[1] - rc[1]
                if abs(dx) >= abs(dy):
                    rel_type = "right_of" if dx >= 0 else "left_of"
                else:
                    rel_type = "below" if dy >= 0 else "above"
                relations.append(self._rel(rel_type, label["id"], best["id"], weight=0.82))

        native_group_rel_map = {
            "annotation_group_id": "same_annotation_group",
            "legend_group_id": "same_legend_group",
            "series_group_id": "same_series_group",
            "axis_group_id": "same_axis_group",
            "tick_group_id": "same_tick_group",
            "table_row_group_id": "same_row",
            "table_column_group_id": "same_column_group",
            "cell_id": "inside_same_cell",
        }
        for element in top_level:
            hints = element.get("structure_hints") or {}
            attachment_hint = str(hints.get("attachment_target_hint") or "").strip()
            if attachment_hint:
                relations.append(self._rel("native_attachment_hint", element["id"], attachment_hint, weight=0.91))
        for group_key, rel_type in native_group_rel_map.items():
            buckets = {}
            for element in top_level:
                hints = element.get("structure_hints") or {}
                hint_groups = hints.get("group_ids") or {}
                gid = str(hint_groups.get(group_key) or "").strip()
                if gid:
                    buckets.setdefault(gid, []).append(element)
            for gid, members in buckets.items():
                if len(members) < 2:
                    continue
                members.sort(key=lambda el: (el["bbox"][1], el["bbox"][0], el["id"]))
                for idx in range(len(members) - 1):
                    relations.append(self._rel(rel_type, members[idx]["id"], members[idx + 1]["id"], weight=0.89))

        for group in groups:
            region_id = group.get("region_id")
            if region_id:
                relations.append(self._rel("inside", group["id"], region_id, weight=1.0))
            for idx, element_id in enumerate(group.get("element_ids") or []):
                if idx > 0:
                    relations.append(self._rel("continues_as", group["element_ids"][idx - 1], element_id, weight=0.9))

        return relations

    def _build_constraints(self, page_data, elements, groups, region_map):
        constraints = []
        for element in elements:
            role = str(element.get("role") or "").strip().lower()
            unit_type = str((element.get("semantic") or {}).get("unit_type") or "").strip().lower()
            bbox = element.get("bbox") or [0, 0, 0, 0]
            region_id = element.get("page_region_id")
            ai_region_id = element.get("ai_region_id")
            region_type = str((region_map.get(region_id) or {}).get("type") or "").strip().lower() if region_id else ""
            ai_region_type = str((region_map.get(ai_region_id) or {}).get("type") or "").strip().lower() if ai_region_id else ""
            params = {
                "max_width_px": max(1.0, float(bbox[2]) - float(bbox[0])),
                "max_height_px": max(1.0, float(bbox[3]) - float(bbox[1])),
                "allow_vertical_expand": False,
                "allow_horizontal_expand": False,
                "preserve_alignment": (element.get("style") or {}).get("align", "left"),
            }
            if element.get("type") == "figure":
                constraints.append(self._constraint("fixed_bbox", element["id"], params=params, priority=100))
                continue
            if ai_region_type in {"header", "footer", "paragraph_title", "title"} and element.get("type") in {
                "text_block", "text_line", "text_phrase", "title", "section_header", "header", "footer"
            }:
                constraints.append(
                    self._constraint("anchored_bbox", element["id"], region_id=ai_region_id or region_id, params=params, priority=93)
                )
                continue
            if ai_region_type in {"text", "paragraph"} and element.get("type") in {"text_block", "text_line", "text_phrase"} and role in {"body", "paragraph", "list_item"}:
                params["allow_vertical_expand"] = True
                constraints.append(
                    self._constraint("flow_in_region", element["id"], region_id=ai_region_id or region_id, params=params, priority=86)
                )
                constraints.append(self._constraint("preserve_visible_text", element["id"], params={"required": True}, priority=92))
                constraints.append(self._constraint("no_internal_sentence_break", element["id"], params={"enabled": True}, priority=88))
                continue
            if region_type in {"table_cell", "table_row"} and element.get("type") in {"text_block", "text_line", "text_phrase"}:
                constraints.append(self._constraint("table_cell_locked", element["id"], region_id=region_id, params=params, priority=97))
                constraints.append(self._constraint("preserve_visible_text", element["id"], params={"required": True}, priority=92))
                continue
            if role in {"diagram_label", "diagram_text_label", "figure_caption", "header", "footer", "title", "section_heading"}:
                constraints.append(self._constraint("anchored_bbox", element["id"], region_id=region_id, params=params, priority=90))
                continue
            if region_type in {"annotation_band", "caption_band", "header_band"} and element.get("type") in {"text_block", "text_line", "text_phrase"}:
                constraints.append(self._constraint("anchored_bbox", element["id"], region_id=region_id, params=params, priority=91))
                continue
            if (
                str(page_data.get("layout_type") or "").strip().lower() == "annotated_page"
                and element.get("type") in {"text_block", "text_line", "text_phrase"}
                and role == "body"
                and unit_type == "narrative_body"
            ):
                word_count = len(re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", str(((element.get("text") or {}).get("source_text") or ""))))
                if word_count >= 12 or region_type == "text_band":
                    params["allow_vertical_expand"] = True
                    constraints.append(self._constraint("flow_in_region", element["id"], region_id=region_id, params=params, priority=84))
                    constraints.append(self._constraint("preserve_visible_text", element["id"], params={"required": True}, priority=92))
                    constraints.append(self._constraint("no_internal_sentence_break", element["id"], params={"enabled": True}, priority=88))
                    continue
            if role in {"equation_inline", "equation_block"} or unit_type in {"formula", "formula_label"}:
                constraints.append(self._constraint("fixed_bbox", element["id"], region_id=region_id, params=params, priority=95))
                continue
            if element.get("type") == "table_cell":
                constraints.append(self._constraint("table_cell_locked", element["id"], region_id=region_id, params=params, priority=95))
                continue
            if element.get("type") in {"text_phrase", "text_line", "text_block"} and role in {"body", "paragraph", "list_item"}:
                params["allow_vertical_expand"] = bool(
                    page_data.get("layout_type") in {"single_column", "double_column", "text_dense"}
                    and unit_type not in {"short_label", "chart_label", "diagram_label", "reference_link", "citation"}
                )
                constraints.append(self._constraint("preserve_visible_text", element["id"], params={"required": True}, priority=92))
                constraints.append(self._constraint("no_internal_sentence_break", element["id"], params={"enabled": True}, priority=88))

        for group in groups:
            params = {
                "can_break_inside_sentence": False,
                "allow_vertical_expand": bool((group.get("constraints") or {}).get("allow_vertical_expand")),
            }
            constraints.append(
                self._constraint(
                    "flow_in_region",
                    group["id"],
                    region_id=group.get("region_id"),
                    params=params,
                    priority=70,
                )
            )
            constraints.append(self._constraint("keep_same_column", group["id"], region_id=group.get("region_id"), params={}, priority=75))
        return constraints

    def _build_reading_order(self, elements):
        top_level = [el["id"] for el in sorted(elements, key=lambda el: (el.get("reading_order", 0), el["id"])) if not el.get("parent_id")]
        return top_level

    def _build_features(self, page_data, elements, groups, regions, page_w, page_h):
        blocks = page_data.get("blocks") or []
        text_elements = [el for el in elements if el.get("type") in {"text_block", "text_line", "text_phrase"}]
        figures = [el for el in elements if el.get("type") == "figure"]
        captions = [el for el in elements if el.get("type") == "caption" or el.get("role") == "figure_caption"]
        formulas = [el for el in elements if el.get("role") in {"equation_inline", "equation_block"}]
        font_sizes = []
        aligns = {}
        for el in text_elements:
            style = el.get("style") or {}
            fs = float(style.get("font_size_px", 0.0) or 0.0)
            if fs > 0:
                font_sizes.append(round(fs, 1))
            aligns[str(style.get("align") or "left")] = aligns.get(str(style.get("align") or "left"), 0) + 1
        total_area = max(1.0, page_w * page_h)
        text_area = sum(self._area(el.get("bbox")) for el in text_elements if el.get("type") == "text_block")
        image_area = sum(self._area(el.get("bbox")) for el in figures)
        formula_area = sum(self._area(el.get("bbox")) for el in formulas)
        table_area = sum(self._area(r.get("bbox")) for r in regions if r.get("type") == "table")
        col_regions = [r for r in regions if r.get("type") == "column"]
        col_areas = [self._area(r.get("bbox")) for r in col_regions]
        column_balance_score = 1.0
        if len(col_areas) >= 2 and max(col_areas) > 0:
            column_balance_score = min(col_areas) / max(col_areas)
        probs = []
        total_align = sum(aligns.values()) or 1
        for count in aligns.values():
            p = count / total_align
            probs.append(p)
        alignment_entropy = -sum(p * math.log(max(p, 1e-9), 2) for p in probs) if probs else 0.0
        return {
            "num_columns": len(col_regions) or max(1, len((page_data.get("layout") or {}).get("columns") or [])),
            "text_coverage_ratio": round(text_area / total_area, 4),
            "table_coverage_ratio": round(table_area / total_area, 4),
            "image_coverage_ratio": round(image_area / total_area, 4),
            "formula_coverage_ratio": round(formula_area / total_area, 4),
            "whitespace_ratio": round(max(0.0, 1.0 - min(1.0, (text_area + table_area + image_area) / total_area)), 4),
            "header_present": any(str(b.get("role") or "").strip().lower() == "header" for b in blocks),
            "footer_present": any(str(b.get("role") or "").strip().lower() == "footer" for b in blocks),
            "title_count": sum(1 for b in blocks if str(b.get("role") or "").strip().lower() in {"title", "section_heading"}),
            "text_block_count": sum(1 for b in blocks if str(b.get("role") or "").strip().lower() in {"body", "paragraph", "list_item"}),
            "table_count": sum(1 for r in regions if r.get("type") == "table"),
            "figure_count": len(figures),
            "caption_count": len(captions),
            "footnote_count": sum(1 for b in blocks if str(b.get("role") or "").strip().lower() == "footnote"),
            "font_size_levels": len(set(font_sizes)),
            "dominant_font_size": max(set(font_sizes), key=font_sizes.count) if font_sizes else 0.0,
            "alignment_entropy": round(alignment_entropy, 4),
            "column_balance_score": round(column_balance_score, 4),
            "toc_pattern_score": float((page_data.get("page_case") or {}).get("features", {}).get("toc_pattern_score", 0.0) or 0.0),
            "form_pattern_score": float((page_data.get("page_case") or {}).get("features", {}).get("form_pattern_score", 0.0) or 0.0),
            "scientific_pattern_score": float((page_data.get("page_case") or {}).get("features", {}).get("scientific_pattern_score", 0.0) or 0.0),
            "invoice_pattern_score": float((page_data.get("page_case") or {}).get("features", {}).get("invoice_pattern_score", 0.0) or 0.0),
            "paragraph_group_count": len(groups),
            "ai_region_count": sum(1 for r in regions if self._is_ai_region(r)),
            "ai_text_region_count": sum(
                1 for r in regions if self._is_ai_region(r) and str(r.get("type") or "").strip().lower() in {"text", "paragraph"}
            ),
        }

    def _enrich_elements_structure(self, page_data, elements, groups, regions, relations, constraints):
        region_map = {r["id"]: r for r in regions}
        relation_targets = {}
        for rel in relations:
            relation_targets.setdefault(rel.get("source_id"), []).append(rel)
        constraint_map = {}
        for constraint in constraints:
            constraint_map.setdefault(constraint.get("element_id"), []).append(constraint)

        elements_by_id = {el.get("id"): el for el in elements}
        heading_to_section = {}
        for rel in relations:
            if rel.get("type") != "heads_content":
                continue
            heading_id = rel.get("source_id")
            heading = elements_by_id.get(heading_id) or {}
            heading_role = str(heading.get("role") or "").strip().lower()
            if heading_role == "header":
                continue
            section_id = f"section_{heading_id}"
            heading_to_section[heading_id] = section_id

        for element in elements:
            page_region = region_map.get(element.get("page_region_id")) or {}
            ai_region = region_map.get(element.get("ai_region_id")) or {}
            rels = relation_targets.get(element.get("id"), [])
            own_constraints = constraint_map.get(element.get("id"), [])
            band_role = self._infer_band_role(page_region, ai_region, element)
            structural_role = self._infer_structural_role(page_data, element, page_region, ai_region, rels)
            layout_behavior = self._infer_layout_behavior(own_constraints, page_region, ai_region, element)
            attachment_target_id = self._infer_attachment_target_id(element, rels)
            group_ids = self._infer_group_ids(element, page_region, ai_region, rels)
            group_render_mode = self._infer_group_render_mode(element, page_region, ai_region, group_ids)
            typographic_class = self._infer_typographic_class(element, page_region, ai_region)
            visual_text_type = self._infer_visual_text_type(element, page_region, ai_region)
            text_embedding_mode = self._infer_text_embedding_mode(element, page_region, ai_region, attachment_target_id, group_render_mode)
            background_kind = self._infer_background_kind(page_region, ai_region, text_embedding_mode)
            background_replacement_strategy = self._infer_background_replacement_strategy(
                text_embedding_mode,
                band_role,
                structural_role,
                background_kind,
            )
            sentence_integrity = {
                "no_internal_break": any(c.get("type") == "no_internal_sentence_break" for c in own_constraints),
                "paragraph_may_reflow": any(c.get("type") == "flow_in_region" for c in own_constraints),
            }

            section_id = None
            if element.get("id") in heading_to_section:
                section_id = heading_to_section[element.get("id")]
            else:
                candidate_section_ids = []
                for rel in relations:
                    if rel.get("type") != "heads_content" or rel.get("target_id") != element.get("id"):
                        continue
                    source_id = rel.get("source_id")
                    source = elements_by_id.get(source_id) or {}
                    source_role = str(source.get("role") or "").strip().lower()
                    if source_role == "header":
                        continue
                    candidate_section_ids.append(
                        (
                            0 if source_role in {"section_heading", "title"} else 1,
                            source.get("reading_order", 0),
                            heading_to_section.get(source_id) or f"section_{source_id}",
                        )
                    )
                if candidate_section_ids:
                    candidate_section_ids.sort()
                    section_id = candidate_section_ids[0][2]

            element["band_role"] = band_role
            element["structural_role"] = structural_role
            element["layout_behavior"] = layout_behavior
            element["attachment_target_id"] = attachment_target_id
            element["section_id"] = section_id
            element["group_ids"] = group_ids
            element["group_render_mode"] = group_render_mode
            element["typographic_class"] = typographic_class
            element["visual_text"] = {
                "type": visual_text_type,
                "text_embedding_mode": text_embedding_mode,
                "background_kind": background_kind,
                "background_replacement_strategy": background_replacement_strategy,
            }
            element["sentence_integrity"] = sentence_integrity

        for group in groups:
            region = region_map.get(group.get("region_id")) or {}
            ai_region = region_map.get(group.get("ai_region_id")) or {}
            group["band_role"] = self._infer_band_role(region, ai_region, None)
            group["layout_behavior"] = "flow_in_region"
            group["group_render_mode"] = "flow_group"
            group["typographic_class"] = "editorial_body"
            group["group_ids"] = {
                "section_id": None,
                "table_id": self._table_id_for_region(region),
                "annotation_group_id": None,
            }

    def _build_native_structure(self, page_data):
        native = page_data.get("native_structure") or {}
        return {
            "table": native.get("table"),
            "annotations": native.get("annotations"),
            "chart": native.get("chart"),
        }

    def _build_page_organization(self, page_data, regions, elements, groups, relations, ai_structure, native_structure):
        region_map = {r["id"]: r for r in regions}
        bands = []
        visual_zones = []
        table_zones = []
        chart_zones = []
        formula_zones = []
        for region in regions:
            r_type = str(region.get("type") or "").strip().lower()
            descriptor = {
                "id": region.get("id"),
                "type": r_type,
                "source": region.get("source"),
                "bbox": region.get("bbox"),
                "column_index": region.get("column_index"),
                "parent_region_id": region.get("parent_region_id"),
            }
            if r_type in {"header_band", "footer_band", "text_band", "caption_band", "annotation_band", "header", "footer", "text", "paragraph_title", "title"}:
                bands.append(descriptor)
            if r_type in {"illustration", "picture", "image", "figure"}:
                visual_zones.append(descriptor)
            if r_type in {"table", "table_row", "table_cell"}:
                table_zones.append(descriptor)
            if r_type in {"chart_area", "chart_plot_area", "chart_y_ticks", "chart_x_ticks", "chart_y_axis", "chart_x_axis", "chart_legend", "chart"}:
                chart_zones.append(descriptor)
            if r_type in {"formula"}:
                formula_zones.append(descriptor)

        sections = []
        for rel in relations:
            if rel.get("type") != "heads_content":
                continue
            heading_id = rel.get("source_id")
            target_id = rel.get("target_id")
            heading = next((el for el in elements if el.get("id") == heading_id), None)
            target = next((el for el in elements if el.get("id") == target_id), None)
            if not heading or not target:
                continue
            sections.append(
                {
                    "id": f"section_{heading_id}",
                    "heading_id": heading_id,
                    "heading_bbox": heading.get("bbox"),
                    "opening_element_id": target_id,
                    "column_index": heading.get("column_index"),
                }
            )

        columns = (page_data.get("layout") or {}).get("columns") or []
        return {
            "reading_mode": self._reading_mode_for_page(page_data),
            "page_density": self._page_density(page_data, elements),
            "columns": columns,
            "bands": bands,
            "visual_zones": visual_zones,
            "table_zones": table_zones,
            "chart_zones": chart_zones,
            "formula_zones": formula_zones,
            "sections": sections,
            "ai_useful_regions": list(ai_structure.get("regions") or []),
            "native_structure": native_structure,
            "annotation_groups": list((native_structure.get("annotations") or {}).get("groups") or []),
            "table_row_groups": list((native_structure.get("table") or {}).get("row_groups") or []),
            "table_header_row_group_ids": list((native_structure.get("table") or {}).get("header_row_group_ids") or []),
            "table_stub_column_group_id": (native_structure.get("table") or {}).get("stub_column_group_id"),
            "chart_groups": {
                "axis_groups": list((native_structure.get("chart") or {}).get("axis_groups") or []),
                "tick_group": (native_structure.get("chart") or {}).get("y_tick_group"),
                "x_tick_group": (native_structure.get("chart") or {}).get("x_tick_group"),
                "legend_group": (native_structure.get("chart") or {}).get("legend_group"),
                "series_groups": list((native_structure.get("chart") or {}).get("series_groups") or []),
                "plot_area_bbox": (native_structure.get("chart") or {}).get("plot_area_bbox"),
            },
            "primary_flow_region_ids": [
                r.get("id") for r in regions if str(r.get("type") or "").strip().lower() in {"text_band", "text", "column"}
            ],
            "anchored_region_ids": [
                r.get("id") for r in regions if str(r.get("type") or "").strip().lower() in {"header_band", "caption_band", "annotation_band", "header", "footer"}
            ],
            "locked_region_ids": [
                r.get("id") for r in regions if str(r.get("type") or "").strip().lower() in {"table", "table_row", "table_cell", "chart_area", "illustration"}
            ],
        }

    def _build_reconstruction_plan(self, page_data, regions, elements, groups, constraints, page_organization):
        keep_out_zones = []
        for region in regions:
            r_type = str(region.get("type") or "").strip().lower()
            if r_type in {"illustration", "picture", "image", "figure", "chart_area", "table"}:
                keep_out_zones.append({"region_id": region.get("id"), "bbox": region.get("bbox"), "type": r_type})
        alignment_guides = []
        for col in page_organization.get("columns") or []:
            alignment_guides.append(
                {
                    "type": "column",
                    "column_index": int(col.get("id", 0) or 0),
                    "x0": float(col.get("x0", 0.0) or 0.0),
                    "x1": float(col.get("x1", 0.0) or 0.0),
                }
            )
        render_sequence = self._render_sequence_for_page(page_data)
        return {
            "render_sequence": render_sequence,
            "primary_flow_regions": list(page_organization.get("primary_flow_region_ids") or []),
            "anchored_regions": list(page_organization.get("anchored_region_ids") or []),
            "locked_regions": list(page_organization.get("locked_region_ids") or []),
            "keep_out_zones": keep_out_zones,
            "alignment_guides": alignment_guides,
            "group_integrity_rules": {
                "paragraph_groups": len(groups),
                "annotation_groups": len((page_organization.get("annotation_groups") or [])),
                "table_row_groups": len((page_organization.get("table_row_groups") or [])),
                "sentence_breaks_forbidden": sum(
                    1 for c in constraints if c.get("type") == "no_internal_sentence_break"
                ),
                "table_locked_elements": sum(
                    1 for c in constraints if c.get("type") == "table_cell_locked"
                ),
            },
        }

    def _build_visual_text_model(self, page_data, elements, page_organization):
        objects = []
        groups = []
        for element in elements:
            visual = element.get("visual_text") or {}
            visual_type = str(visual.get("type") or "").strip().lower()
            if not visual_type or visual_type in {"body_text", "editorial_heading"}:
                continue
            if element.get("parent_id"):
                continue
            group_ids = element.get("group_ids") or {}
            group_id = (
                group_ids.get("annotation_group_id")
                or group_ids.get("legend_group_id")
                or group_ids.get("axis_group_id")
                or group_ids.get("tick_group_id")
                or group_ids.get("series_group_id")
            )
            embedding_mode = str(visual.get("text_embedding_mode") or "").strip().lower()
            replacement_strategy = str(visual.get("background_replacement_strategy") or "").strip().lower()
            background_kind = str(visual.get("background_kind") or "").strip().lower()
            visual_priority = "secondary"
            if visual_type in {"diagram_title", "chart_title", "chart_axis_label"}:
                visual_priority = "primary"
            elif visual_type in {"chart_tick_label", "visual_micro_label"}:
                visual_priority = "micro"
            objects.append(
                {
                    "visual_text_id": f"vt_{element.get('id')}",
                    "source_element_id": element.get("id"),
                    "source_block_id": element.get("id").split("_ln_", 1)[0] if isinstance(element.get("id"), str) else element.get("id"),
                    "page_id": int(page_data.get("page", 1) or 1) - 1,
                    "type": visual_type,
                    "semantic_type": visual_type,
                    "text_embedding_mode": embedding_mode,
                    "background_kind": background_kind,
                    "background_replacement_strategy": replacement_strategy,
                    "bbox": element.get("bbox"),
                    "tight_bbox": element.get("bbox"),
                    "anchor_point": element.get("center"),
                    "baseline": element.get("baseline"),
                    "rotation": 0.0,
                    "orientation": "horizontal",
                    "local_padding": [2.0, 1.5, 2.0, 1.5],
                    "group_render_mode": element.get("group_render_mode"),
                    "attachment_target_id": element.get("attachment_target_id"),
                    "band_role": element.get("band_role"),
                    "structural_role": element.get("structural_role"),
                    "visual_priority": visual_priority,
                    "visual_parent_id": element.get("attachment_target_id"),
                    "contrast_requirement": "high" if background_kind in {"diagram_fill", "chart_fill", "mixed_texture"} else "normal",
                    "must_preserve_relative_position": embedding_mode != "outside_visual",
                    "must_preserve_alignment": visual_priority != "micro",
                    "allow_reflow": False,
                    "allow_multiline": visual_type not in {"chart_tick_label", "visual_micro_label"},
                    "erase_source_text": replacement_strategy == "text_erase_then_overlay",
                    "needs_background_estimation": replacement_strategy == "text_erase_then_overlay",
                    "background_patch_source": "estimated_local_patch" if replacement_strategy == "text_erase_then_overlay" else ("source_crop" if replacement_strategy == "crop_restore" else "whiteout"),
                    "visibility_priority": "must_preserve_visible_text",
                    "must_not_duplicate_source_text": embedding_mode in {"overlay_on_visual", "embedded_in_visual"},
                    "group_ids": group_ids,
                    "group_id": group_id,
                }
            )
        seen = set()
        for obj in objects:
            gids = obj.get("group_ids") or {}
            group_id = obj.get("group_id")
            if not group_id or group_id in seen:
                continue
            seen.add(group_id)
            members = [o for o in objects if ((o.get("group_ids") or {}).get("annotation_group_id") == group_id or (o.get("group_ids") or {}).get("legend_group_id") == group_id or (o.get("group_ids") or {}).get("axis_group_id") == group_id or (o.get("group_ids") or {}).get("tick_group_id") == group_id or (o.get("group_ids") or {}).get("series_group_id") == group_id)]
            if not members:
                continue
            bbox = [
                min(m["bbox"][0] for m in members),
                min(m["bbox"][1] for m in members),
                max(m["bbox"][2] for m in members),
                max(m["bbox"][3] for m in members),
            ]
            groups.append(
                {
                    "id": group_id,
                    "render_mode": members[0].get("group_render_mode"),
                    "text_embedding_mode": members[0].get("text_embedding_mode"),
                    "background_replacement_strategy": members[0].get("background_replacement_strategy"),
                    "background_kind": members[0].get("background_kind"),
                    "bbox": bbox,
                    "member_ids": [m["visual_text_id"] for m in members],
                    "visual_parent_id": members[0].get("visual_parent_id"),
                    "must_preserve_relative_position": True,
                    "must_not_duplicate_source_text": any(bool(m.get("must_not_duplicate_source_text")) for m in members),
                }
            )
        render_plan = {
            "embedded_objects": [o["visual_text_id"] for o in objects if o.get("text_embedding_mode") == "embedded_in_visual"],
            "overlay_objects": [o["visual_text_id"] for o in objects if o.get("text_embedding_mode") == "overlay_on_visual"],
            "outside_objects": [o["visual_text_id"] for o in objects if o.get("text_embedding_mode") == "outside_visual"],
            "erase_then_overlay_group_ids": [g["id"] for g in groups if g.get("background_replacement_strategy") == "text_erase_then_overlay"],
            "crop_restore_group_ids": [g["id"] for g in groups if g.get("background_replacement_strategy") == "crop_restore"],
        }
        return {
            "objects": objects,
            "groups": groups,
            "render_plan": render_plan,
            "group_count": len(groups),
            "object_count": len(objects),
            "chart_group_count": len(page_organization.get("chart_groups") or {}),
            "annotation_group_count": len(page_organization.get("annotation_groups") or []),
        }

    def _build_ai_structure(self, page_data, regions, elements, relations):
        ai_regions = [r for r in regions if self._is_ai_region(r)]
        useful_regions = [r for r in ai_regions if self._is_high_value_ai_region(r, page_data)]
        useful_ids = {r.get("id") for r in useful_regions}
        top_level = [el for el in elements if not el.get("parent_id")]
        element_links = []
        for element in top_level:
            ai_region_id = element.get("ai_region_id")
            if not ai_region_id or ai_region_id not in useful_ids:
                continue
            region = next((r for r in useful_regions if r.get("id") == ai_region_id), None)
            if not region:
                continue
            element_links.append(
                {
                    "element_id": element.get("id"),
                    "element_role": element.get("role"),
                    "element_type": element.get("type"),
                    "ai_region_id": ai_region_id,
                    "ai_region_type": region.get("type"),
                    "complementary_role": self._ai_complementary_role(region),
                }
            )

        structural_links = [
            rel for rel in relations
            if rel.get("type") in {"inside_ai_region", "same_structural_band", "title_of_region", "heads_content"}
        ]
        return {
            "enabled": bool(ai_regions),
            "raw_region_count": len(ai_regions),
            "useful_region_count": len(useful_regions),
            "regions": [
                {
                    "id": region.get("id"),
                    "type": region.get("type"),
                    "bbox": region.get("bbox"),
                    "score": region.get("score"),
                    "complementary_role": self._ai_complementary_role(region),
                }
                for region in useful_regions
            ],
            "element_links": element_links,
            "structural_links": structural_links,
        }

    def _element_type_for_role(self, role):
        role = str(role or "").strip().lower()
        mapping = {
            "body": "text_block",
            "paragraph": "text_block",
            "list_item": "text_block",
            "title": "title",
            "section_heading": "section_header",
            "header": "header",
            "footer": "footer",
            "figure_caption": "caption",
            "diagram_label": "caption",
            "diagram_text_label": "caption",
            "equation_inline": "formula",
            "equation_block": "formula",
            "footnote": "footnote",
            "page_number": "page_number",
        }
        return mapping.get(role, "text_block")

    def _style_payload(self, style):
        if not isinstance(style, dict):
            style = {}
        flags = style.get("flags") or {}
        font_size = style.get("size") or style.get("font_size") or style.get("font_size_px") or 0.0
        return {
            "font_family": str(style.get("font") or style.get("font_family") or ""),
            "font_size_px": float(font_size or 0.0),
            "font_weight": 700 if bool(flags.get("bold")) else 400,
            "italic": bool(flags.get("italic")),
            "underline": bool(flags.get("underline")),
            "color": str(style.get("color") or "#000000"),
            "align": str(style.get("align") or style.get("alignment") or "left"),
            "line_height_px": float(style.get("line_height") or 0.0),
        }

    def _text_payload(self, node, visible_text):
        if not isinstance(node, dict):
            node = {}
        source_text = self._clean_text(node.get("text") or node.get("line_text") or node.get("texte") or visible_text or "")
        translated_text = node.get("translated_text")
        translated_text = self._clean_text(translated_text) if translated_text else None
        text = visible_text or translated_text or source_text
        return {
            "source_text": source_text,
            "visible_text": text,
            "translated_text": translated_text,
            "language": str(node.get("language") or ""),
            "tokens": len(re.findall(r"[A-Za-zÀ-ÿ0-9]+", text or "")),
            "is_truncated_source": bool(node.get("visible_text") and source_text and source_text != node.get("visible_text")),
        }

    def _semantic_payload(self, node, block_role, block=None):
        base = block if isinstance(block, dict) else {}
        merged = {}
        if isinstance(base, dict):
            merged.update(base)
        if isinstance(node, dict):
            merged.update(node)
        unit_type = str(merged.get("unit_type") or self._default_unit_type(block_role)).strip().lower()
        source_text = self._clean_text(
            merged.get("translated_text") or merged.get("text") or merged.get("line_text") or merged.get("texte") or ""
        )
        formula_markers = ("=", "+", "-", "*", "/", "^", "max(", "min(")
        return {
            "unit_type": unit_type,
            "is_translatable": unit_type not in {"figure", "formula", "page_number"},
            "is_reference_like": unit_type in {"reference_link", "citation"} or bool(re.search(r"https?://|doi:|www\\.", source_text, flags=re.I)),
            "is_code_like": bool(re.search(r"\\bif\\b|\\bfor\\b|\\bwhile\\b|\\{\\}|\\(\\)|==|<=|>=|:=|def\\s+", source_text)),
            "is_formula_like": unit_type in {"formula", "formula_label"} or any(marker in source_text for marker in formula_markers),
        }

    def _default_unit_type(self, role):
        role = str(role or "").strip().lower()
        if role in {"diagram_label", "diagram_text_label"}:
            return "diagram_label"
        if role == "figure_caption":
            return "caption"
        if role in {"equation_inline", "equation_block"}:
            return "formula"
        if role in {"title", "section_heading"}:
            return "heading"
        return "narrative_body" if role in {"body", "paragraph", "list_item"} else role or "unknown"

    def _line_style(self, block, line):
        for phrase in (line.get("phrases") or []):
            for span in (phrase.get("spans") or []):
                if isinstance(span.get("style"), dict):
                    return span.get("style")
        return line.get("style") or block.get("style") or {}

    def _phrase_style(self, block, phrase):
        for span in (phrase.get("spans") or []):
            if isinstance(span.get("style"), dict):
                return span.get("style")
        return phrase.get("style") or block.get("style") or {}

    def _best_region_id(self, bbox, regions):
        best = None
        best_area = -1.0
        best_region_area = None
        for region in regions:
            rb = region.get("bbox") or [0, 0, 0, 0]
            area = self._intersection_area(bbox, rb)
            region_area = self._area(rb)
            if area > best_area or (
                area == best_area and area > 0 and (best_region_area is None or region_area < best_region_area)
            ):
                best_area = area
                best_region_area = region_area
                best = region.get("id")
        return best

    def _best_ai_region_id(self, bbox, regions):
        ai_regions = [region for region in regions if self._is_ai_region(region)]
        if not ai_regions:
            return None
        return self._best_region_id(bbox, ai_regions)

    def _is_high_value_ai_region(self, region, page_data):
        if not self._is_ai_region(region):
            return False
        r_type = str(region.get("type") or "").strip().lower()
        score = float(region.get("score", 0.0) or 0.0)
        layout_type = str(page_data.get("layout_type") or "").strip().lower()
        document_type = str(page_data.get("document_type") or "").strip().lower()
        if r_type in {"supplementaryregion", "number"}:
            return False
        if r_type == "formula":
            return score >= 0.55 and (
                layout_type in {"table_dominant", "annotated_page", "mixed_blocks"}
                or document_type in {"scientific_paper", "form"}
            )
        if layout_type == "annotated_page":
            return r_type in {"text", "image", "chart", "header", "paragraph_title", "figure_title", "caption", "table"}
        if layout_type == "table_dominant":
            return r_type in {"table", "text", "header", "paragraph_title", "image", "caption", "formula"}
        return r_type in {"text", "image", "table", "chart", "header", "paragraph_title", "title", "caption", "figure_title", "formula"}

    def _ai_complementary_role(self, region):
        r_type = str((region or {}).get("type") or "").strip().lower()
        mapping = {
            "text": "text_band",
            "paragraph_title": "title_band",
            "title": "title_band",
            "header": "header_band",
            "footer": "footer_band",
            "image": "visual_zone",
            "picture": "visual_zone",
            "figure": "visual_zone",
            "table": "table_envelope",
            "chart": "chart_envelope",
            "caption": "caption_band",
            "figure_title": "caption_band",
            "formula": "formula_zone",
        }
        return mapping.get(r_type, "structural_hint")

    def _infer_band_role(self, page_region, ai_region, element):
        hints = (element or {}).get("structure_hints") or {}
        hinted_band = str(hints.get("band_role_hint") or "").strip()
        if hinted_band:
            return hinted_band
        region_type = str((page_region or {}).get("type") or "").strip().lower()
        ai_type = str((ai_region or {}).get("type") or "").strip().lower()
        for candidate in (region_type, ai_type):
            mapping = {
                "header_band": "header_band",
                "header": "header_band",
                "footer_band": "footer_band",
                "footer": "footer_band",
                "text_band": "text_band",
                "text": "text_band",
                "annotation_band": "annotation_band",
                "caption_band": "caption_band",
                "paragraph_title": "title_band",
                "title": "title_band",
                "chart_legend": "legend_band",
                "chart_x_axis": "axis_band",
                "chart_y_axis": "axis_band",
                "chart_y_ticks": "axis_band",
                "chart_x_ticks": "axis_band",
                "table": "table_band",
                "table_row": "table_band",
                "table_cell": "table_band",
            }
            if candidate in mapping:
                return mapping[candidate]
        if element and element.get("role") in {"header", "footer"}:
            return f"{element.get('role')}_band"
        return "content_band"

    def _infer_structural_role(self, page_data, element, page_region, ai_region, rels):
        hints = (element or {}).get("structure_hints") or {}
        hinted_role = str(hints.get("structural_role_hint") or "").strip()
        if hinted_role:
            return hinted_role
        role = str(element.get("role") or "").strip().lower()
        unit_type = str((element.get("semantic") or {}).get("unit_type") or "").strip().lower()
        region_type = str((page_region or {}).get("type") or "").strip().lower()
        ai_type = str((ai_region or {}).get("type") or "").strip().lower()
        if role == "header":
            return "running_header"
        if role == "footer":
            return "running_footer"
        if role == "figure_caption":
            return "figure_caption"
        if role == "section_heading":
            return "section_title"
        if role == "title" and ai_type in {"paragraph_title", "title"}:
            return "section_title"
        if unit_type in {"chart_label"}:
            if region_type == "chart_legend" or ai_type == "chart":
                return "chart_legend_label"
            return "chart_label"
        if region_type == "chart_y_ticks":
            return "chart_tick_label"
        if region_type == "chart_x_ticks":
            return "chart_tick_label"
        if region_type in {"chart_x_axis", "chart_y_axis"}:
            return "chart_axis_label"
        if region_type == "chart_legend":
            return "chart_legend_label"
        hints = (element or {}).get("structure_hints") or {}
        if str(hints.get("structural_role_hint") or "").strip().lower() == "table_stub_cell":
            return "table_stub_cell"
        if region_type == "table_cell":
            if role in {"title", "section_heading"}:
                return "table_header_cell"
            return "table_value_cell"
        if role in {"diagram_label", "diagram_text_label"} or unit_type == "diagram_label":
            return "diagram_label"
        if ai_type == "figure_title":
            return "figure_title"
        if role in {"equation_inline", "equation_block"} or unit_type in {"formula", "formula_label"}:
            return "formula_block"
        if role in {"body", "paragraph", "list_item"}:
            if any(rel.get("type") == "heads_content" for rel in rels):
                return "opening_paragraph"
            if element.get("paragraph_id"):
                return "body_paragraph"
        if role == "title":
            return "label"
        return role or "content"

    def _infer_layout_behavior(self, own_constraints, page_region, ai_region, element):
        hints = (element or {}).get("structure_hints") or {}
        hinted_behavior = str(hints.get("layout_behavior_hint") or "").strip()
        if hinted_behavior:
            return hinted_behavior
        constraint_types = [c.get("type") for c in own_constraints]
        if "table_cell_locked" in constraint_types:
            return "locked_in_cell"
        if "fixed_bbox" in constraint_types:
            return "fixed"
        if "anchored_bbox" in constraint_types:
            return "anchored"
        if "flow_in_region" in constraint_types:
            return "flow_in_band"
        region_type = str((page_region or {}).get("type") or "").strip().lower()
        ai_type = str((ai_region or {}).get("type") or "").strip().lower()
        if region_type in {"table", "table_row", "table_cell"}:
            return "locked_in_table"
        if ai_type in {"header", "paragraph_title"}:
            return "anchored"
        if element and element.get("type") == "figure":
            return "fixed"
        return "flow"

    def _infer_attachment_target_id(self, element, rels):
        hints = (element or {}).get("structure_hints") or {}
        hinted_target = str(hints.get("attachment_target_hint") or "").strip()
        if hinted_target:
            return hinted_target
        # native structure hint is stronger than weak relational fallback
        for rel in rels:
            if rel.get("type") == "native_attachment_hint" and rel.get("target_id"):
                return rel.get("target_id")
        for rel_type in ("anchored_to", "caption_of", "title_of_region", "inside_ai_region"):
            for rel in rels:
                if rel.get("type") == rel_type:
                    return rel.get("target_id")
        return None

    def _infer_group_ids(self, element, page_region, ai_region, rels):
        region_type = str((page_region or {}).get("type") or "").strip().lower()
        ai_type = str((ai_region or {}).get("type") or "").strip().lower()
        hints = (element or {}).get("structure_hints") or {}
        group_ids = {
            "annotation_group_id": None,
            "legend_group_id": None,
            "series_group_id": None,
            "axis_group_id": None,
            "tick_group_id": None,
            "table_id": None,
            "table_row_group_id": None,
            "table_column_group_id": None,
            "cell_id": None,
        }
        hinted_group_ids = hints.get("group_ids") or {}
        for key in group_ids:
            if hinted_group_ids.get(key):
                group_ids[key] = hinted_group_ids.get(key)
        if region_type == "chart_legend":
            group_ids["legend_group_id"] = "legend_group_0"
        if region_type in {"chart_x_axis", "chart_y_axis"}:
            group_ids["axis_group_id"] = f"axis_group_{region_type}"
        if region_type == "chart_y_ticks":
            group_ids["tick_group_id"] = "tick_group_y"
        if region_type == "table":
            group_ids["table_id"] = page_region.get("id")
        if region_type == "table_row":
            group_ids["table_id"] = self._table_id_for_region(page_region)
            group_ids["table_row_group_id"] = page_region.get("id")
        if region_type == "table_cell":
            group_ids["table_id"] = self._table_id_for_region(page_region)
            group_ids["table_row_group_id"] = str((page_region or {}).get("parent_region_id") or "") or None
            group_ids["cell_id"] = page_region.get("id")
        if any(rel.get("type") == "anchored_to" for rel in rels):
            target = next((rel.get("target_id") for rel in rels if rel.get("type") == "anchored_to"), None)
            if target and not group_ids["annotation_group_id"]:
                group_ids["annotation_group_id"] = f"annotation_group_{target}"
        if ai_type == "chart":
            group_ids["legend_group_id"] = group_ids["legend_group_id"] or "chart_group_0"
        return group_ids

    def _infer_group_render_mode(self, element, page_region, ai_region, group_ids):
        hints = (element or {}).get("structure_hints") or {}
        hinted = str(hints.get("group_render_mode_hint") or "").strip().lower()
        if hinted:
            return hinted
        structural_role = str((element or {}).get("structural_role") or (element or {}).get("role") or "").strip().lower()
        band_role = str((element or {}).get("band_role") or "").strip().lower()
        region_type = str((page_region or {}).get("type") or "").strip().lower()
        ai_type = str((ai_region or {}).get("type") or "").strip().lower()
        if group_ids.get("annotation_group_id") or band_role == "annotation_band" or structural_role == "diagram_label":
            return "annotation_group"
        if group_ids.get("legend_group_id") or structural_role == "chart_legend_label" or region_type == "chart_legend":
            return "chart_legend_group"
        if (
            group_ids.get("axis_group_id")
            or group_ids.get("tick_group_id")
            or structural_role in {"chart_axis_label", "chart_tick_label"}
            or region_type in {"chart_x_axis", "chart_y_axis", "chart_x_ticks", "chart_y_ticks"}
        ):
            return "chart_axis_group"
        if group_ids.get("series_group_id") or ai_type == "chart":
            return "chart_series_group"
        if group_ids.get("cell_id") or group_ids.get("table_row_group_id"):
            return "table_group"
        return "flow_group"

    def _infer_typographic_class(self, element, page_region, ai_region):
        structural_role = str((element or {}).get("structural_role") or (element or {}).get("role") or "").strip().lower()
        role = str((element or {}).get("role") or "").strip().lower()
        region_type = str((page_region or {}).get("type") or "").strip().lower()
        ai_type = str((ai_region or {}).get("type") or "").strip().lower()
        if structural_role in {"running_header", "running_footer"} or role in {"header", "footer"}:
            return "running_header"
        if structural_role in {"section_title"} or role in {"section_heading"}:
            return "section_heading"
        if structural_role in {"opening_paragraph", "body_paragraph", "continuation_paragraph"} or role in {"body", "paragraph", "list_item"}:
            return "editorial_body"
        if structural_role in {"figure_caption"} or role == "figure_caption":
            return "figure_caption"
        if structural_role in {"figure_title", "diagram_title"} or (role == "title" and region_type in {"annotation_band", "illustration"}):
            return "diagram_title"
        if structural_role in {"diagram_label"}:
            return "diagram_label"
        if structural_role in {"chart_axis_label"} or region_type in {"chart_x_axis", "chart_y_axis"}:
            return "chart_axis_label"
        if structural_role in {"chart_tick_label"} or region_type in {"chart_x_ticks", "chart_y_ticks"}:
            return "chart_tick_label"
        if structural_role in {"chart_legend_label", "chart_series_label"} or region_type == "chart_legend" or ai_type == "chart":
            return "chart_legend_label"
        if structural_role in {"table_header_cell"}:
            return "table_header_cell"
        if structural_role in {"table_stub_cell"}:
            return "table_stub_cell"
        if structural_role in {"table_value_cell"} or region_type == "table_cell":
            return "table_value_cell"
        return "content"

    def _infer_visual_text_type(self, element, page_region, ai_region):
        structural_role = str((element or {}).get("structural_role") or (element or {}).get("role") or "").strip().lower()
        role = str((element or {}).get("role") or "").strip().lower()
        region_type = str((page_region or {}).get("type") or "").strip().lower()
        ai_type = str((ai_region or {}).get("type") or "").strip().lower()
        if structural_role in {"chart_axis_label", "chart_tick_label", "chart_legend_label", "chart_series_label"}:
            return structural_role
        if structural_role == "diagram_label":
            if role == "title":
                return "diagram_title"
            return "diagram_explanatory_label"
        if role == "figure_caption":
            return "figure_caption"
        if role == "title" and region_type in {"annotation_band", "illustration"}:
            return "diagram_title"
        if role == "title" and ai_type == "chart":
            return "chart_title"
        if role in {"header", "section_heading"}:
            return "editorial_heading"
        if role == "body":
            return "body_text"
        return "visual_micro_label"

    def _infer_text_embedding_mode(self, element, page_region, ai_region, attachment_target_id, group_render_mode):
        band_role = str((element or {}).get("band_role") or "").strip().lower()
        region_type = str((page_region or {}).get("type") or "").strip().lower()
        ai_type = str((ai_region or {}).get("type") or "").strip().lower()
        if band_role == "text_band":
            return "outside_visual"
        if group_render_mode in {"annotation_group", "chart_axis_group", "chart_legend_group", "chart_series_group"}:
            return "embedded_in_visual"
        if attachment_target_id in {"illustration_main", "chart_main"}:
            return "embedded_in_visual"
        if region_type in {"annotation_band", "chart_area", "chart_plot_area", "chart_x_axis", "chart_y_axis", "chart_x_ticks", "chart_y_ticks", "chart_legend"}:
            return "embedded_in_visual"
        if ai_type in {"image", "chart", "figure"}:
            return "overlay_on_visual"
        return "outside_visual"

    def _infer_background_kind(self, page_region, ai_region, text_embedding_mode):
        region_type = str((page_region or {}).get("type") or "").strip().lower()
        ai_type = str((ai_region or {}).get("type") or "").strip().lower()
        if text_embedding_mode == "outside_visual":
            return "plain"
        if region_type in {"chart_area", "chart_plot_area", "chart_x_axis", "chart_y_axis", "chart_x_ticks", "chart_y_ticks", "chart_legend"}:
            return "chart_fill"
        if region_type in {"annotation_band", "illustration"} or ai_type in {"image", "figure"}:
            return "diagram_fill"
        if ai_type == "chart":
            return "chart_fill"
        return "mixed_texture"

    def _infer_background_replacement_strategy(self, text_embedding_mode, band_role, structural_role, background_kind):
        if text_embedding_mode == "outside_visual":
            if band_role == "text_band" or structural_role in {"opening_paragraph", "body_paragraph", "continuation_paragraph"}:
                return "line_whiteout"
            return "whiteout"
        if text_embedding_mode == "overlay_on_visual":
            return "crop_restore"
        if text_embedding_mode == "embedded_in_visual":
            if background_kind in {"diagram_fill", "chart_fill", "mixed_texture"}:
                return "text_erase_then_overlay"
            return "crop_restore"
        return "whiteout"

    def _table_id_for_region(self, region):
        if not isinstance(region, dict):
            return None
        r_type = str(region.get("type") or "").strip().lower()
        if r_type == "table":
            return region.get("id")
        parent = str(region.get("parent_region_id") or "").strip()
        if parent.startswith("region_table_main"):
            return parent
        if parent.startswith("region_table_row_"):
            return "region_table_main"
        return None

    def _reading_mode_for_page(self, page_data):
        layout_type = str(page_data.get("layout_type") or "").strip().lower()
        if layout_type == "double_column":
            return "double_column_flow"
        if layout_type == "table_dominant":
            return "table_first"
        if layout_type == "annotated_page":
            return "visual_then_caption"
        return "single_flow"

    def _page_density(self, page_data, elements):
        dims = page_data.get("dimensions") or {}
        area = max(1.0, float(dims.get("width", 0.0) or 0.0) * float(dims.get("height", 0.0) or 0.0))
        text_blocks = [el for el in elements if el.get("type") == "text_block"]
        coverage = sum(self._area(el.get("bbox")) for el in text_blocks) / area
        if coverage >= 0.28:
            return "dense"
        if coverage >= 0.14:
            return "normal"
        return "sparse"

    def _render_sequence_for_page(self, page_data):
        layout_type = str(page_data.get("layout_type") or "").strip().lower()
        if layout_type == "annotated_page":
            return ["header_footer", "titles", "visual_zones", "captions", "body_flow", "labels"]
        if layout_type == "table_dominant":
            return ["header_footer", "titles", "tables", "formulas", "body_flow", "captions"]
        if layout_type == "double_column":
            return ["header_footer", "titles", "body_flow", "captions", "references"]
        return ["header_footer", "titles", "body_flow", "captions", "labels"]

    def _assign_region_id(self, page_data, block, bbox, regions):
        default_region = self._best_region_id(bbox, regions)
        layout_type = str(page_data.get("layout_type") or "").strip().lower()
        if layout_type != "annotated_page":
            return default_region
        role = str(block.get("role") or "").strip().lower()
        unit_type = str(block.get("unit_type") or self._default_unit_type(role)).strip().lower()
        text = self._clean_text(block.get("translated_text") or block.get("text") or "")
        words = len(re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", text))
        candidate_types = []
        if role == "figure_caption":
            candidate_types = ["caption_band", "illustration"]
        elif role in {"header", "title", "section_heading"} and bbox[1] <= float(page_data.get("dimensions", {}).get("height", 0.0) or 0.0) * 0.18:
            candidate_types = ["header_band", "annotation_band", "text_band"]
        elif role == "body" and words >= 12:
            candidate_types = ["text_band"]
        elif role in {"title", "diagram_label", "diagram_text_label", "section_heading"} or unit_type in {"short_label", "chart_label", "diagram_label"}:
            candidate_types = ["annotation_band", "illustration"]
        if not candidate_types:
            return default_region
        best = None
        best_score = None
        for region in regions:
            region_type = str(region.get("type") or "").strip().lower()
            if region_type not in candidate_types:
                continue
            rb = region.get("bbox") or [0, 0, 0, 0]
            overlap = self._intersection_area(bbox, rb)
            dist = self._bbox_distance(bbox, rb)
            score = overlap * 5.0 - dist
            if best_score is None or score > best_score:
                best = region.get("id")
                best_score = score
        return best or default_region

    def _is_ai_region(self, region):
        if not isinstance(region, dict):
            return False
        source = str(region.get("source") or "").strip().lower()
        region_id = str(region.get("id") or "").strip().lower()
        return source == "layout_ai" or region_id.startswith("ai_region_")

    def _column_index_for_bbox(self, bbox, columns):
        if not bbox or not columns:
            return 0
        cx = (float(bbox[0]) + float(bbox[2])) / 2.0
        best = 0
        best_dist = None
        for col in columns:
            x0 = float(col.get("x0", 0.0) or 0.0)
            x1 = float(col.get("x1", x0) or x0)
            if x0 <= cx <= x1:
                return int(col.get("id", best))
            dist = min(abs(cx - x0), abs(cx - x1))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = int(col.get("id", best))
        return best

    def _clean_text(self, text):
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def _norm_bbox(self, bbox):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            x0, y0, x1, y1 = [float(v) for v in bbox]
        except Exception:
            return None
        if x1 <= x0 or y1 <= y0:
            return None
        return [x0, y0, x1, y1]

    def _center(self, bbox):
        if not bbox:
            return [0.0, 0.0]
        return [(float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0]

    def _area(self, bbox):
        if not bbox:
            return 0.0
        return max(0.0, (float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1])))

    def _intersection_area(self, a, b):
        if not a or not b:
            return 0.0
        x0 = max(float(a[0]), float(b[0]))
        y0 = max(float(a[1]), float(b[1]))
        x1 = min(float(a[2]), float(b[2]))
        y1 = min(float(a[3]), float(b[3]))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return (x1 - x0) * (y1 - y0)

    def _bbox_distance(self, a, b):
        if not a or not b:
            return 1e9
        x_gap = max(0.0, max(float(b[0]) - float(a[2]), float(a[0]) - float(b[2])))
        y_gap = max(0.0, max(float(b[1]) - float(a[3]), float(a[1]) - float(b[3])))
        if x_gap == 0.0 and y_gap == 0.0:
            return 0.0
        return math.hypot(x_gap, y_gap)

    def _rel(self, rel_type, source_id, target_id, weight=1.0, metadata=None):
        return {
            "id": f"rel_{rel_type}_{source_id}_{target_id}",
            "type": rel_type,
            "source_id": source_id,
            "target_id": target_id,
            "weight": float(weight),
            "metadata": metadata or {},
        }

    def _constraint(self, c_type, element_id, region_id=None, params=None, priority=50):
        data = {
            "id": f"c_{c_type}_{element_id}",
            "type": c_type,
            "element_id": element_id,
            "params": params or {},
            "priority": int(priority),
        }
        if region_id:
            data["region_id"] = region_id
        return data
