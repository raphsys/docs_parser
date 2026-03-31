import os
import copy


class LayoutAIEnricher:
    ENABLE_ENV = "LAYOUT_AI_ENABLE"
    LOCAL_MODEL_ENV = "LAYOUT_AI_MODELS_ROOT"
    PROFILE_ENV = "LAYOUT_AI_PROFILE"
    DEFAULT_MODELS_ROOT = os.path.join(os.path.dirname(__file__), "ai_models", "ppstructurev3")
    MINIMAL_MODELS = {
        "layout_detection_model_dir": "PP-DocLayout_plus-L",
        "chart_recognition_model_dir": "PP-Chart2Table",
        "region_detection_model_dir": "PP-DocBlockLayout",
        "text_detection_model_dir": "PP-OCRv5_server_det",
        "textline_orientation_model_dir": "PP-LCNet_x1_0_textline_ori",
        "text_recognition_model_dir": "PP-OCRv5_server_rec",
    }
    DOC_PREPROCESSOR_MODELS = {
        "doc_orientation_model_dir": "PP-LCNet_x1_0_doc_ori",
        "doc_unwarping_model_dir": "UVDoc",
    }
    TABLE_MODELS = {
        "table_classification_model_dir": "PP-LCNet_x1_0_table_cls",
        "wired_table_structure_model_dir": "SLANeXt_wired",
        "wireless_table_structure_model_dir": "SLANet_plus",
        "wired_table_cells_model_dir": "RT-DETR-L_wired_table_cell_det",
        "wireless_table_cells_model_dir": "RT-DETR-L_wireless_table_cell_det",
    }
    FORMULA_MODELS = {
        "formula_recognition_model_dir": "PP-FormulaNet_plus-L",
    }
    SEAL_MODELS = {
        "seal_detection_model_dir": "PP-OCRv4_server_seal_det",
    }
    ADVANCED_MODELS = {
        **DOC_PREPROCESSOR_MODELS,
        **TABLE_MODELS,
        **FORMULA_MODELS,
        **SEAL_MODELS,
    }
    MODEL_PATHS = {
        "layout_detection_model_dir": ("SubModules", "LayoutDetection", "model_dir"),
        "chart_recognition_model_dir": ("SubModules", "ChartRecognition", "model_dir"),
        "region_detection_model_dir": ("SubModules", "RegionDetection", "model_dir"),
        "doc_orientation_model_dir": ("SubPipelines", "DocPreprocessor", "SubModules", "DocOrientationClassify", "model_dir"),
        "doc_unwarping_model_dir": ("SubPipelines", "DocPreprocessor", "SubModules", "DocUnwarping", "model_dir"),
        "text_detection_model_dir": ("SubPipelines", "GeneralOCR", "SubModules", "TextDetection", "model_dir"),
        "textline_orientation_model_dir": ("SubPipelines", "GeneralOCR", "SubModules", "TextLineOrientation", "model_dir"),
        "text_recognition_model_dir": ("SubPipelines", "GeneralOCR", "SubModules", "TextRecognition", "model_dir"),
        "table_classification_model_dir": ("SubPipelines", "TableRecognition", "SubModules", "TableClassification", "model_dir"),
        "wired_table_structure_model_dir": ("SubPipelines", "TableRecognition", "SubModules", "WiredTableStructureRecognition", "model_dir"),
        "wireless_table_structure_model_dir": ("SubPipelines", "TableRecognition", "SubModules", "WirelessTableStructureRecognition", "model_dir"),
        "wired_table_cells_model_dir": ("SubPipelines", "TableRecognition", "SubModules", "WiredTableCellsDetection", "model_dir"),
        "wireless_table_cells_model_dir": ("SubPipelines", "TableRecognition", "SubModules", "WirelessTableCellsDetection", "model_dir"),
        "seal_detection_model_dir": ("SubPipelines", "SealRecognition", "SubPipelines", "SealOCR", "SubModules", "TextDetection", "model_dir"),
        "formula_recognition_model_dir": ("SubPipelines", "FormulaRecognition", "SubModules", "FormulaRecognition", "model_dir"),
    }
    PARSING_LABEL_MAP = {
        "doc_title": "title",
        "title": "title",
        "paragraph_title": "paragraph_title",
        "figure_title": "figure_title",
        "header": "header",
        "footer": "footer",
        "text": "text",
        "reference": "text",
        "abstract": "text",
        "content": "text",
        "list": "list_item",
        "caption": "caption",
        "figure_caption": "caption",
        "table": "table",
        "formula": "formula",
        "chart": "chart",
        "image": "image",
        "figure": "image",
        "picture": "image",
        "seal": "seal",
    }
    MAX_INFERENCE_SIDE = 1320

    def __init__(self):
        self.backend = str(os.getenv("LAYOUT_AI_BACKEND", "ppstructurev3") or "ppstructurev3").strip().lower()
        self.models_root = str(os.getenv(self.LOCAL_MODEL_ENV, self.DEFAULT_MODELS_ROOT) or self.DEFAULT_MODELS_ROOT).strip()
        requested_profile = str(os.getenv(self.PROFILE_ENV, "auto") or "auto").strip().lower()
        self.profile = self._resolve_profile(requested_profile)
        enabled_env = os.getenv(self.ENABLE_ENV)
        if enabled_env is None or str(enabled_env).strip() == "":
            self.enabled = self._has_local_model_bundle(self.MINIMAL_MODELS)
        else:
            self.enabled = str(enabled_env).strip() == "1"
        self.feature_flags = self._build_feature_flags()
        self._pipeline = None
        self._load_error = None

    def status(self):
        return {
            "enabled": bool(self.enabled),
            "backend": self.backend,
            "profile": self.profile,
            "models_root": self.models_root,
            "minimal_models_present": self._has_local_model_bundle(self.MINIMAL_MODELS),
            "advanced_models_present": self._has_local_model_bundle(self.ADVANCED_MODELS),
            "feature_flags": dict(self.feature_flags),
            "ready": self._pipeline is not None,
            "load_error": self._load_error,
        }

    def enrich(self, page_data, pil_img):
        if not isinstance(page_data, dict):
            return page_data, {"enabled": self.enabled, "ready": False, "applied": False}
        info = {
            "enabled": bool(self.enabled),
            "backend": self.backend,
            "profile": self.profile,
            "ready": False,
            "applied": False,
            "regions_added": 0,
            "feature_flags": dict(self.feature_flags),
            "prediction_summary": {},
            "load_error": None,
            "predict_error": None,
            "inference_rescaled": False,
        }
        if not self.enabled:
            return page_data, info

        pipeline = self._get_pipeline()
        info["ready"] = pipeline is not None
        info["load_error"] = self._load_error
        if pipeline is None or pil_img is None:
            return page_data, info

        infer_img, scale_meta = self._prepare_inference_image(pil_img)
        info["inference_rescaled"] = bool(scale_meta.get("rescaled"))
        predictions = self._predict(pipeline, infer_img)
        if predictions is None:
            info["predict_error"] = self._load_error
            return page_data, info

        prediction_summary = self._summarize_predictions(predictions)
        if prediction_summary:
            page_data["layout_ai_outputs"] = prediction_summary
            info["prediction_summary"] = prediction_summary
        structural_payload = self._extract_structural_payload(predictions)
        if scale_meta.get("rescaled"):
            structural_payload = self._rescale_structural_payload(
                structural_payload,
                scale_x=float(scale_meta.get("scale_x", 1.0) or 1.0),
                scale_y=float(scale_meta.get("scale_y", 1.0) or 1.0),
            )
        if structural_payload:
            page_data["layout_ai_structure"] = structural_payload
        regions = list(structural_payload.get("regions") or [])
        if not regions:
            regions = self._extract_regions(predictions)
        if not regions:
            return page_data, info

        existing = list(page_data.get("ai_layout_regions") or [])
        page_data["ai_layout_regions"] = existing + regions
        info["applied"] = True
        info["regions_added"] = len(regions)
        return page_data, info

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        if self._load_error is not None:
            return None
        if self.backend != "ppstructurev3":
            self._load_error = f"unsupported_backend:{self.backend}"
            return None
        try:
            from paddlex.inference.pipelines import create_pipeline, load_pipeline_config
            from paddlex.inference.utils.pp_option import PaddlePredictorOption
        except Exception as exc:
            self._load_error = f"import_error:{type(exc).__name__}:{exc}"
            return None
        try:
            cfg = load_pipeline_config("PP-StructureV3")
            cfg = copy.deepcopy(cfg)
            cfg["use_doc_preprocessor"] = self.feature_flags["doc_preprocessor"]
            cfg["use_seal_recognition"] = self.feature_flags["seal_recognition"]
            cfg["use_table_recognition"] = self.feature_flags["table_recognition"]
            cfg["use_formula_recognition"] = self.feature_flags["formula_recognition"]
            cfg["use_chart_recognition"] = self.feature_flags["chart_recognition"]
            cfg["use_region_detection"] = self.feature_flags["region_detection"]
            cfg["format_block_content"] = False
            self._apply_local_model_dirs(cfg)

            pp_option = PaddlePredictorOption()
            pp_option.device_type = "cpu"
            pp_option.run_mode = "paddle"
            pp_option.enable_new_ir = False
            pp_option.mkldnn_cache_capacity = 0

            os.environ.setdefault("FLAGS_use_mkldnn", "0")
            os.environ.setdefault("FLAGS_enable_pir_api", "0")
            os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

            self._pipeline = create_pipeline(
                config=cfg,
                device="cpu",
                pp_option=pp_option,
                use_hpip=False,
            )
        except Exception as exc:
            self._load_error = f"init_error:{type(exc).__name__}:{exc}"
            self._pipeline = None
        return self._pipeline

    def _apply_local_model_dirs(self, cfg):
        root = self.models_root
        if not root:
            return
        for arg_name, model_dir_name in {**self.MINIMAL_MODELS, **self.ADVANCED_MODELS}.items():
            local_dir = os.path.join(root, model_dir_name)
            if os.path.isdir(local_dir):
                path = self.MODEL_PATHS.get(arg_name)
                if not path:
                    continue
                ref = cfg
                for key in path[:-1]:
                    ref = ref.setdefault(key, {})
                ref[path[-1]] = local_dir

    def _resolve_profile(self, requested_profile):
        profile = requested_profile if requested_profile in {"auto", "minimal", "advanced"} else "auto"
        if profile == "advanced":
            return "advanced"
        if profile == "minimal":
            return "minimal"
        if self._has_local_model_bundle(self.MINIMAL_MODELS) and self._has_local_model_bundle(self.ADVANCED_MODELS):
            return "advanced"
        return "minimal"

    def _has_local_model_bundle(self, model_map):
        if not self.models_root:
            return False
        for model_dir_name in model_map.values():
            if not os.path.isdir(os.path.join(self.models_root, model_dir_name)):
                return False
        return True

    def _build_feature_flags(self):
        advanced_requested = self.profile == "advanced"
        chart_ready = os.path.isdir(os.path.join(self.models_root, self.MINIMAL_MODELS["chart_recognition_model_dir"]))
        return {
            "doc_preprocessor": advanced_requested and self._has_local_model_bundle(self.DOC_PREPROCESSOR_MODELS),
            "seal_recognition": advanced_requested and self._has_local_model_bundle(self.SEAL_MODELS),
            "table_recognition": advanced_requested and self._has_local_model_bundle(self.TABLE_MODELS),
            "formula_recognition": advanced_requested and self._has_local_model_bundle(self.FORMULA_MODELS),
            "chart_recognition": advanced_requested and chart_ready,
            "region_detection": True,
        }

    def _predict(self, pipeline, pil_img):
        try:
            import numpy as np
        except Exception as exc:
            self._load_error = f"numpy_error:{type(exc).__name__}:{exc}"
            return None
        image = np.array(pil_img)
        try:
            if hasattr(pipeline, "predict"):
                result = pipeline.predict(image)
                if hasattr(result, "__iter__") and not isinstance(result, (dict, list, tuple, str, bytes)):
                    return next(iter(result), None)
                return result
            return pipeline(image)
        except Exception as exc:
            self._load_error = f"predict_error:{type(exc).__name__}:{exc}"
            return None

    def _extract_regions(self, predictions):
        return list((self._extract_structural_payload(predictions) or {}).get("regions") or [])

    def _extract_structural_payload(self, predictions):
        if predictions is None:
            return {}
        if hasattr(predictions, "items"):
            predictions = dict(predictions)
        if not isinstance(predictions, dict):
            regions = self._normalize_prediction_regions(predictions, source="layout_ai")
            return {"regions": regions}

        parsing_blocks = self._normalize_parsing_blocks(predictions.get("parsing_res_list"))
        table_regions = self._normalize_region_collection(predictions.get("table_res_list"), default_type="table", source="layout_ai_table")
        formula_regions = self._normalize_region_collection(predictions.get("formula_res_list"), default_type="formula", source="layout_ai_formula")
        chart_regions = self._normalize_region_collection(predictions.get("chart_res_list"), default_type="chart", source="layout_ai_chart")
        seal_regions = self._normalize_region_collection(predictions.get("seal_res_list"), default_type="seal", source="layout_ai_seal")
        ocr_lines = self._normalize_overall_ocr(predictions.get("overall_ocr_res"))

        regions = []
        regions.extend(self._normalize_prediction_regions((predictions.get("layout_det_res") or {}).get("boxes"), source="layout_ai"))
        regions.extend(self._normalize_prediction_regions((predictions.get("region_det_res") or {}).get("boxes"), source="layout_ai"))
        for idx, item in enumerate(parsing_blocks):
            if not item.get("bbox"):
                continue
            regions.append(
                {
                    "id": item.get("id") or f"ai_parsing_region_{idx}",
                    "type": self.PARSING_LABEL_MAP.get(str(item.get("label") or "").strip().lower(), str(item.get("label") or "unknown").strip().lower() or "unknown"),
                    "bbox": list(item.get("bbox") or []),
                    "score": float(item.get("score", 0.0) or 0.0),
                    "source": "layout_ai_parsing",
                    "label": item.get("label"),
                    "text": item.get("text"),
                }
            )
        for collection in (table_regions, formula_regions, chart_regions, seal_regions):
            regions.extend(collection)
        regions = self._dedupe_regions(regions)

        return {
            "regions": regions,
            "parsing_blocks": parsing_blocks,
            "table_regions": table_regions,
            "formula_regions": formula_regions,
            "chart_regions": chart_regions,
            "seal_regions": seal_regions,
            "ocr_lines": ocr_lines,
        }

    def _normalize_predictions(self, predictions):
        if predictions is None:
            return []
        if hasattr(predictions, "items"):
            predictions = dict(predictions)
        if isinstance(predictions, list):
            flat = []
            for item in predictions:
                if isinstance(item, list):
                    flat.extend(item)
                else:
                    flat.append(item)
            return flat
        if isinstance(predictions, dict):
            payload = self._extract_structural_payload(predictions)
            regions = list(payload.get("regions") or [])
            if self._safe_len(regions):
                return regions
            for key in ("layout", "regions", "res"):
                items = self._as_list(predictions.get(key))
                if self._safe_len(items):
                    return items
            return []
        try:
            return list(predictions)
        except Exception:
            return []

    def _extract_bbox(self, item):
        item = self._to_plain_data(item)
        if not isinstance(item, dict):
            return None
        bbox = item.get("bbox") or item.get("box") or item.get("coordinate")
        if hasattr(bbox, "tolist") and not isinstance(bbox, (dict, list, tuple)):
            try:
                bbox = bbox.tolist()
            except Exception:
                pass
        if isinstance(bbox, dict):
            vals = [bbox.get("x0"), bbox.get("y0"), bbox.get("x1"), bbox.get("y1")]
            if all(v is not None for v in vals):
                return [int(round(float(v))) for v in vals]
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return [int(round(float(v))) for v in bbox]
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 8:
            xs = [float(bbox[i]) for i in range(0, len(bbox), 2)]
            ys = [float(bbox[i]) for i in range(1, len(bbox), 2)]
            return [int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys)))]
        return None

    def _extract_label(self, item):
        item = self._to_plain_data(item)
        if not isinstance(item, dict):
            return "unknown"
        for key in ("label", "type", "category", "cls_name", "layout_type"):
            val = item.get(key)
            if val:
                return str(val).strip().lower()
        return "unknown"

    def _extract_score(self, item):
        item = self._to_plain_data(item)
        if not isinstance(item, dict):
            return 0.0
        for key in ("score", "confidence", "prob"):
            val = item.get(key)
            if val is not None:
                try:
                    return float(val)
                except Exception:
                    return 0.0
        return 0.0

    def _summarize_predictions(self, predictions):
        if hasattr(predictions, "items"):
            predictions = dict(predictions)
        if not isinstance(predictions, dict):
            return {}
        layout_det_res = predictions.get("layout_det_res")
        region_det_res = predictions.get("region_det_res")
        overall_ocr_res = predictions.get("overall_ocr_res")
        payload = self._extract_structural_payload(predictions)
        return {
            "layout_boxes": self._safe_len(layout_det_res.get("boxes")) if isinstance(layout_det_res, dict) else 0,
            "region_boxes": self._safe_len(region_det_res.get("boxes")) if isinstance(region_det_res, dict) else 0,
            "table_results": self._safe_len(predictions.get("table_res_list")),
            "formula_results": self._safe_len(predictions.get("formula_res_list")),
            "seal_results": self._safe_len(predictions.get("seal_res_list")),
            "parsing_results": self._safe_len(predictions.get("parsing_res_list")),
            "overall_ocr_boxes": self._safe_len(overall_ocr_res.get("rec_boxes")) if isinstance(overall_ocr_res, dict) else 0,
            "structural_regions": self._safe_len(payload.get("regions")),
            "parsing_blocks": self._safe_len(payload.get("parsing_blocks")),
            "ocr_lines": self._safe_len(payload.get("ocr_lines")),
        }

    def _normalize_prediction_regions(self, items, source="layout_ai", default_type=None):
        regions = []
        for idx, item in enumerate(self._as_list(items)):
            plain = self._to_plain_data(item)
            bbox = self._extract_bbox(plain)
            if not bbox:
                continue
            label = default_type or self._extract_label(plain)
            if label in self.PARSING_LABEL_MAP:
                label = self.PARSING_LABEL_MAP[label]
            regions.append(
                {
                    "id": str(plain.get("id") or f"{source}_{idx}"),
                    "type": str(label or "unknown").strip().lower() or "unknown",
                    "bbox": bbox,
                    "score": self._extract_score(plain),
                    "source": source,
                    "text": str(plain.get("text") or plain.get("content") or "").strip(),
                }
            )
        return regions

    def _normalize_parsing_blocks(self, items):
        out = []
        for idx, item in enumerate(self._as_list(items)):
            plain = self._to_plain_data(item)
            bbox = self._extract_bbox(plain)
            if not bbox:
                continue
            label = str(plain.get("label") or plain.get("type") or "unknown").strip().lower() or "unknown"
            text = str(plain.get("content") or plain.get("text") or "").strip()
            out.append(
                {
                    "id": str(plain.get("id") or f"ai_parsing_block_{idx}"),
                    "label": label,
                    "bbox": bbox,
                    "text": text,
                    "score": self._extract_score(plain),
                    "order_index": self._safe_number(plain.get("order_index"), default=idx),
                    "num_lines": self._safe_number(plain.get("num_of_lines"), default=0),
                    "direction": str(plain.get("direction") or "").strip().lower(),
                    "text_line_height": self._safe_float(plain.get("text_line_height"), default=0.0),
                    "text_line_width": self._safe_float(plain.get("text_line_width"), default=0.0),
                    "block_height": self._safe_float(plain.get("height"), default=0.0),
                    "block_width": self._safe_float(plain.get("width"), default=0.0),
                }
            )
        return out

    def _normalize_region_collection(self, items, default_type, source):
        return self._normalize_prediction_regions(items, source=source, default_type=default_type)

    def _normalize_overall_ocr(self, overall_ocr_res):
        plain = self._to_plain_data(overall_ocr_res)
        if not isinstance(plain, dict):
            return []
        rec_texts = self._as_list(plain.get("rec_texts"))
        rec_boxes = self._as_list(plain.get("rec_boxes"))
        rec_scores = self._as_list(plain.get("rec_scores"))
        out = []
        for idx, text in enumerate(rec_texts):
            bbox = None
            if idx < len(rec_boxes):
                bbox = self._extract_bbox({"bbox": rec_boxes[idx]})
            if not bbox:
                continue
            score = 0.0
            if idx < len(rec_scores):
                try:
                    score = float(rec_scores[idx])
                except Exception:
                    score = 0.0
            out.append(
                {
                    "id": f"ai_ocr_line_{idx}",
                    "text": str(text or "").strip(),
                    "bbox": bbox,
                    "score": score,
                }
            )
        return out

    def _to_plain_data(self, value, depth=0):
        if depth > 6:
            return None
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {k: self._to_plain_data(v, depth + 1) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_plain_data(v, depth + 1) for v in value]
        if hasattr(value, "to_dict") and callable(value.to_dict):
            try:
                return self._to_plain_data(value.to_dict(), depth + 1)
            except Exception:
                pass
        if hasattr(value, "tolist"):
            try:
                return self._to_plain_data(value.tolist(), depth + 1)
            except Exception:
                pass
        if hasattr(value, "item") and callable(value.item):
            try:
                return value.item()
            except Exception:
                pass
        attrs = {}
        for name in ("label", "bbox", "content", "score", "order_index", "num_of_lines", "direction"):
            if hasattr(value, name):
                try:
                    attrs[name] = self._to_plain_data(getattr(value, name), depth + 1)
                except Exception:
                    continue
        return attrs or str(value)

    def _dedupe_regions(self, regions):
        out = []
        seen = set()
        for region in regions or []:
            bbox = tuple(region.get("bbox") or [])
            key = (str(region.get("type") or ""), bbox, str(region.get("source") or ""))
            if key in seen:
                continue
            seen.add(key)
            out.append(region)
        return out

    def _prepare_inference_image(self, pil_img):
        if pil_img is None or not hasattr(pil_img, "size"):
            return pil_img, {"rescaled": False, "scale_x": 1.0, "scale_y": 1.0}
        try:
            width, height = pil_img.size
        except Exception:
            return pil_img, {"rescaled": False, "scale_x": 1.0, "scale_y": 1.0}
        max_side = max(int(width or 0), int(height or 0))
        if max_side <= self.MAX_INFERENCE_SIDE or max_side <= 0:
            return pil_img, {"rescaled": False, "scale_x": 1.0, "scale_y": 1.0}
        ratio = float(self.MAX_INFERENCE_SIDE) / float(max_side)
        new_w = max(32, int(round(width * ratio)))
        new_h = max(32, int(round(height * ratio)))
        resized = pil_img.resize((new_w, new_h))
        return resized, {
            "rescaled": True,
            "scale_x": float(width) / float(new_w),
            "scale_y": float(height) / float(new_h),
        }

    def _rescale_structural_payload(self, payload, scale_x=1.0, scale_y=1.0):
        if not isinstance(payload, dict):
            return payload
        out = copy.deepcopy(payload)
        for key in ("regions", "parsing_blocks", "table_regions", "formula_regions", "chart_regions", "seal_regions", "ocr_lines"):
            items = out.get(key) or []
            for item in items:
                bbox = self._extract_bbox(item)
                if bbox:
                    item["bbox"] = self._scale_bbox(bbox, scale_x=scale_x, scale_y=scale_y)
                if key == "parsing_blocks":
                    if item.get("text_line_height"):
                        item["text_line_height"] = float(item.get("text_line_height") or 0.0) * scale_y
                    if item.get("text_line_width"):
                        item["text_line_width"] = float(item.get("text_line_width") or 0.0) * scale_x
                    if item.get("block_height"):
                        item["block_height"] = float(item.get("block_height") or 0.0) * scale_y
                    if item.get("block_width"):
                        item["block_width"] = float(item.get("block_width") or 0.0) * scale_x
        return out

    def _scale_bbox(self, bbox, scale_x=1.0, scale_y=1.0):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return bbox
        try:
            x0, y0, x1, y1 = [float(v) for v in bbox]
        except Exception:
            return bbox
        return [
            int(round(x0 * scale_x)),
            int(round(y0 * scale_y)),
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
        ]

    def _as_list(self, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        try:
            return list(value)
        except Exception:
            return []

    def _safe_len(self, value):
        if value is None:
            return 0
        try:
            return len(value)
        except Exception:
            return 0

    def _safe_number(self, value, default=0):
        try:
            return int(value)
        except Exception:
            try:
                return float(value)
            except Exception:
                return default

    def _safe_float(self, value, default=0.0):
        try:
            return float(value)
        except Exception:
            return default


_INSTANCE = None


def get_layout_ai_enricher():
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = LayoutAIEnricher()
    return _INSTANCE
