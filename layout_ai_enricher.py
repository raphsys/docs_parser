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
        }
        if not self.enabled:
            return page_data, info

        pipeline = self._get_pipeline()
        info["ready"] = pipeline is not None
        info["load_error"] = self._load_error
        if pipeline is None or pil_img is None:
            return page_data, info

        predictions = self._predict(pipeline, pil_img)
        if predictions is None:
            info["predict_error"] = self._load_error
            return page_data, info

        prediction_summary = self._summarize_predictions(predictions)
        if prediction_summary:
            page_data["layout_ai_outputs"] = prediction_summary
            info["prediction_summary"] = prediction_summary
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
        regions = []
        items = self._normalize_predictions(predictions)
        for idx, item in enumerate(items):
            bbox = self._extract_bbox(item)
            if not bbox:
                continue
            label = self._extract_label(item)
            score = self._extract_score(item)
            regions.append(
                {
                    "id": f"ai_region_{idx}",
                    "type": label,
                    "bbox": bbox,
                    "score": score,
                    "source": "layout_ai",
                }
            )
        return regions

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
            layout_boxes = ((predictions.get("layout_det_res") or {}).get("boxes") or [])
            region_boxes = ((predictions.get("region_det_res") or {}).get("boxes") or [])
            if layout_boxes or region_boxes:
                return list(layout_boxes) + list(region_boxes)
            return predictions.get("layout") or predictions.get("regions") or predictions.get("res") or []
        try:
            return list(predictions)
        except Exception:
            return []

    def _extract_bbox(self, item):
        if not isinstance(item, dict):
            return None
        bbox = item.get("bbox") or item.get("box") or item.get("coordinate")
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
        if not isinstance(item, dict):
            return "unknown"
        for key in ("label", "type", "category", "cls_name", "layout_type"):
            val = item.get(key)
            if val:
                return str(val).strip().lower()
        return "unknown"

    def _extract_score(self, item):
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
        return {
            "layout_boxes": len(((predictions.get("layout_det_res") or {}).get("boxes") or [])),
            "region_boxes": len(((predictions.get("region_det_res") or {}).get("boxes") or [])),
            "table_results": len(predictions.get("table_res_list") or []),
            "formula_results": len(predictions.get("formula_res_list") or []),
            "seal_results": len(predictions.get("seal_res_list") or []),
            "parsing_results": len(predictions.get("parsing_res_list") or []),
            "overall_ocr_boxes": len(((predictions.get("overall_ocr_res") or {}).get("rec_boxes") or [])),
        }


_INSTANCE = None


def get_layout_ai_enricher():
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = LayoutAIEnricher()
    return _INSTANCE
