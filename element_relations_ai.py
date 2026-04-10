import os

import numpy as np


SCHEMA_VERSION = "element_relations_ai.v1"


class ElementRelationsAIEnricher:
    ENABLE_ENV = "ELEMENT_RELATIONS_AI_ENABLE"
    MODEL_DIR_ENV = "ELEMENT_RELATIONS_AI_MODEL_DIR"
    BACKEND_ENV = "ELEMENT_RELATIONS_AI_BACKEND"
    MIN_CONF_ENV = "ELEMENT_RELATIONS_AI_MIN_CONFIDENCE"
    DEFAULT_BACKEND = "onnx_nli"
    DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "ai_models", "element_relations_nli")
    MODEL_FILENAMES = (
        "model.onnx",
        "model_quint8_avx2.onnx",
        "model_qint8_arm64.onnx",
        "model_qint8_avx512.onnx",
        "model_qint8_avx512_vnni.onnx",
        "model_O4.onnx",
        "model_O3.onnx",
        "model_O2.onnx",
        "model_O1.onnx",
    )
    CONTINUATION_HYPOTHESES = {
        "continuation": "the second fragment is a continuation of the previous text",
        "new_unit": "the second fragment starts a new textual unit",
    }
    LOGICAL_HYPOTHESES = {
        "same_token_continuation": "the second fragment continues the same broken token",
        "same_sentence_continuation": "the second fragment continues the same sentence on the same line",
        "same_paragraph_continuation": "the second fragment continues the same paragraph after a line wrap",
        "new_list_item": "the second fragment starts a new list item",
        "new_sentence_or_unit": "the second fragment starts a new sentence or a new unit",
        "new_structural_unit": "the second fragment starts a new structural unit",
    }

    def __init__(self):
        self.backend = str(os.getenv(self.BACKEND_ENV, self.DEFAULT_BACKEND) or self.DEFAULT_BACKEND).strip().lower()
        self.model_dir = str(os.getenv(self.MODEL_DIR_ENV, self.DEFAULT_MODEL_DIR) or self.DEFAULT_MODEL_DIR).strip()
        self.min_confidence = self._safe_float(os.getenv(self.MIN_CONF_ENV), 0.78)
        enabled_env = os.getenv(self.ENABLE_ENV)
        if enabled_env is None or str(enabled_env).strip() == "":
            self.enabled = self._has_local_model_bundle()
        else:
            self.enabled = str(enabled_env).strip() == "1"
        self._runtime = None
        self._load_error = None
        self._score_cache = {}

    def status(self):
        return {
            "enabled": bool(self.enabled),
            "backend": self.backend,
            "model_dir": self.model_dir,
            "ready": self._runtime is not None,
            "load_error": self._load_error,
            "min_confidence": self.min_confidence,
            "local_model_present": self._has_local_model_bundle(),
        }

    def score_hypotheses(self, premise, hypotheses):
        runtime = self._get_runtime()
        if runtime is None:
            return {}
        cache_key = self._score_cache_key(premise, hypotheses)
        if cache_key in self._score_cache:
            return dict(self._score_cache[cache_key])
        scores = self._score_hypotheses(premise, hypotheses)
        self._score_cache[cache_key] = dict(scores)
        return dict(scores)

    def enrich(self, page_data):
        info = {
            "schema_version": SCHEMA_VERSION,
            "enabled": bool(self.enabled),
            "backend": self.backend,
            "model_dir": self.model_dir,
            "ready": False,
            "applied": False,
            "load_error": None,
            "candidate_relations": 0,
            "reviewed_relations": 0,
            "resolved_relations": 0,
            "min_confidence": self.min_confidence,
        }
        if not isinstance(page_data, dict):
            return page_data, info

        page_data.setdefault("layout", {})
        payload = page_data.get("element_relations") or {}
        flat_relations = [
            relation
            for relation in (payload.get("flat_relations") or [])
            if isinstance(relation, dict)
        ]

        candidates = [
            relation
            for relation in flat_relations
            if bool(relation.get("ai_review_required")) or str(relation.get("logical_relation") or "") == "uncertain"
        ]
        info["candidate_relations"] = len(candidates)

        if not self.enabled or not candidates:
            self._attach_info(page_data, info)
            return page_data, info

        runtime = self._get_runtime()
        info["ready"] = runtime is not None
        info["load_error"] = self._load_error
        if runtime is None:
            self._attach_info(page_data, info)
            return page_data, info

        for relation in candidates:
            review = self._review_relation(relation)
            if not review:
                continue
            info["reviewed_relations"] += 1
            if self._merge_relation_review(relation, review):
                info["resolved_relations"] += 1

        info["applied"] = info["reviewed_relations"] > 0
        self._attach_info(page_data, info)
        return page_data, info

    def _attach_info(self, page_data, info):
        page_data["element_relations_ai"] = info
        page_data.setdefault("layout", {})
        page_data["layout"]["element_relations_ai"] = info
        page_data["layout"]["element_relations_ai_version"] = SCHEMA_VERSION

    def _review_relation(self, relation):
        source_text = str(((relation.get("text") or {}).get("source")) or "").strip()
        target_text = str(((relation.get("text") or {}).get("target")) or "").strip()
        if not source_text or not target_text:
            return None

        premise = self._build_premise(relation, source_text, target_text)
        continuation_scores = self._score_hypotheses(premise, self.CONTINUATION_HYPOTHESES)
        logical_scores = self._score_hypotheses(premise, self.LOGICAL_HYPOTHESES)
        if not continuation_scores or not logical_scores:
            return None

        best_cont_label, best_cont_score = max(continuation_scores.items(), key=lambda item: item[1])
        best_logical_label, best_logical_score = max(logical_scores.items(), key=lambda item: item[1])
        return {
            "review_mode": "onnx_nli",
            "continuation_label": best_cont_label,
            "continuation_confidence": round(float(best_cont_score), 4),
            "logical_label": best_logical_label,
            "logical_confidence": round(float(best_logical_score), 4),
            "continuation_scores": {key: round(float(value), 4) for key, value in continuation_scores.items()},
            "logical_scores": {key: round(float(value), 4) for key, value in logical_scores.items()},
        }

    def _merge_relation_review(self, relation, review):
        if not isinstance(relation, dict) or not isinstance(review, dict):
            return False

        relation.setdefault(
            "heuristic_decision",
            {
                "visual_relation": relation.get("visual_relation"),
                "logical_relation": relation.get("logical_relation"),
                "continuation": relation.get("continuation"),
                "confidence": relation.get("confidence"),
                "ai_review_required": relation.get("ai_review_required"),
            },
        )

        continuation_conf = float(review.get("continuation_confidence") or 0.0)
        logical_conf = float(review.get("logical_confidence") or 0.0)
        continuation_label = str(review.get("continuation_label") or "").strip()
        logical_label = str(review.get("logical_label") or "").strip()
        signals = relation.get("signals") or {}
        same_line = bool(signals.get("same_line"))
        line_delta = int(signals.get("line_delta") or 0)
        prev_terminal = bool(signals.get("previous_terminal_punctuation"))
        curr_hard_break = bool(signals.get("current_hard_break_before"))
        curr_marker = bool(signals.get("current_has_list_marker"))

        resolved = False
        if continuation_conf >= self.min_confidence:
            continuation = continuation_label == "continuation"
            relation["continuation"] = continuation
            if continuation:
                relation["visual_relation"] = "continues_inline" if same_line else "continues_wrapped_line"
            else:
                relation["visual_relation"] = "new_structural_unit"
            resolved = True
        elif (
            continuation_label == "continuation"
            and line_delta in {0, 1}
            and continuation_conf >= 0.6
            and not prev_terminal
            and not curr_hard_break
            and not curr_marker
        ):
            relation["continuation"] = True
            relation["visual_relation"] = "continues_inline" if same_line else "continues_wrapped_line"
            resolved = True

        if logical_conf >= self.min_confidence or str(relation.get("logical_relation") or "") == "uncertain":
            relation["logical_relation"] = logical_label or relation.get("logical_relation")
            resolved = True

        relation["confidence"] = round(max(float(relation.get("confidence") or 0.0), continuation_conf, logical_conf), 4)
        relation["ai_review_required"] = not resolved
        relation["resolved_by"] = "semantic_ai" if resolved else "heuristics"
        relation["semantic_ai_review"] = {
            "applied": True,
            "schema_version": SCHEMA_VERSION,
            "backend": self.backend,
            "model_dir": self.model_dir,
            "review_mode": review.get("review_mode") or self.backend,
            "continuation_label": continuation_label,
            "continuation_confidence": round(continuation_conf, 4),
            "logical_label": logical_label,
            "logical_confidence": round(logical_conf, 4),
            "continuation_scores": dict(review.get("continuation_scores") or {}),
            "logical_scores": dict(review.get("logical_scores") or {}),
        }
        return resolved

    def _build_premise(self, relation, source_text, target_text):
        signals = relation.get("signals") or {}
        return (
            f"Previous fragment: {source_text}\n"
            f"Next fragment: {target_text}\n"
            f"same_line={int(bool(signals.get('same_line')))}; "
            f"line_delta={int(signals.get('line_delta', 0) or 0)}; "
            f"inline_gap_px={float(signals.get('inline_gap_px', 0.0) or 0.0):.2f}; "
            f"vertical_gap_px={float(signals.get('vertical_gap_px', 0.0) or 0.0):.2f}; "
            f"indent_delta_px={float(signals.get('indent_delta_px', 0.0) or 0.0):.2f}; "
            f"previous_terminal_punctuation={int(bool(signals.get('previous_terminal_punctuation')))}; "
            f"previous_ends_hyphen={int(bool(signals.get('previous_ends_hyphen')))}; "
            f"current_starts_lowercase={int(bool(signals.get('current_starts_lowercase')))}; "
            f"current_has_list_marker={int(bool(signals.get('current_has_list_marker')))}; "
            f"current_hard_break_before={int(bool(signals.get('current_hard_break_before')))}."
        )

    def _score_hypotheses(self, premise, hypotheses):
        runtime = self._get_runtime()
        if runtime is None:
            return {}
        batched = self._score_hypotheses_batch(premise, hypotheses, runtime)
        if batched:
            return self._normalize_scores(batched)
        scores = {}
        for label, hypothesis in (hypotheses or {}).items():
            try:
                scores[label] = float(self._score_entailment(premise, hypothesis, runtime))
            except Exception:
                continue
        return self._normalize_scores(scores)

    def _score_hypotheses_batch(self, premise, hypotheses, runtime):
        hypothesis_items = [
            (str(label), str(hypothesis))
            for label, hypothesis in (hypotheses or {}).items()
            if str(label).strip() and str(hypothesis).strip()
        ]
        if not hypothesis_items:
            return {}
        tokenizer = runtime["tokenizer"]
        session = runtime["session"]
        entailment_id = runtime["entailment_id"]
        contradiction_id = runtime["contradiction_id"]
        try:
            encoded = tokenizer(
                [premise] * len(hypothesis_items),
                [hypothesis for _, hypothesis in hypothesis_items],
                return_tensors="np",
                truncation=True,
                padding=True,
            )
            feed = {}
            for input_meta in session.get_inputs():
                name = input_meta.name
                if name in encoded:
                    feed[name] = np.asarray(encoded[name])
            if not feed:
                return {}
            outputs = session.run(None, feed)
            if not outputs:
                return {}
            logits = np.asarray(outputs[0])
            if logits.ndim == 1:
                logits = np.expand_dims(logits, axis=0)
            scores = {}
            for (label, _), row in zip(hypothesis_items, logits):
                probs = self._softmax(row)
                if contradiction_id is not None:
                    denom = float(probs[entailment_id] + probs[contradiction_id]) or 1.0
                    scores[label] = float(probs[entailment_id] / denom)
                else:
                    scores[label] = float(probs[entailment_id])
            return scores
        except Exception:
            return {}

    def _score_entailment(self, premise, hypothesis, runtime):
        tokenizer = runtime["tokenizer"]
        session = runtime["session"]
        entailment_id = runtime["entailment_id"]
        contradiction_id = runtime["contradiction_id"]

        encoded = tokenizer(
            premise,
            hypothesis,
            return_tensors="np",
            truncation=True,
        )
        feed = {}
        for input_meta in session.get_inputs():
            name = input_meta.name
            if name in encoded:
                feed[name] = np.asarray(encoded[name])
        if not feed:
            raise RuntimeError("missing_onnx_inputs")

        outputs = session.run(None, feed)
        if not outputs:
            raise RuntimeError("missing_onnx_outputs")
        logits = np.asarray(outputs[0])[0]
        probs = self._softmax(logits)
        if contradiction_id is not None:
            denom = float(probs[entailment_id] + probs[contradiction_id]) or 1.0
            return float(probs[entailment_id] / denom)
        return float(probs[entailment_id])

    def _get_runtime(self):
        if self._runtime is not None:
            return self._runtime
        if self._load_error is not None:
            return None
        if self.backend != "onnx_nli":
            self._load_error = f"unsupported_backend:{self.backend}"
            return None
        model_path = self._resolve_model_path()
        if not model_path:
            self._load_error = "missing_local_model_bundle"
            return None
        try:
            import onnxruntime as ort
            from transformers import AutoConfig, AutoTokenizer
        except Exception as exc:
            self._load_error = f"import_error:{type(exc).__name__}:{exc}"
            return None
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True, use_fast=True)
            config = AutoConfig.from_pretrained(self.model_dir, local_files_only=True)
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            label_map = self._config_label_map(config)
            entailment_id = self._find_label_id(label_map, "entail")
            contradiction_id = self._find_label_id(label_map, "contrad")
            if entailment_id is None:
                raise RuntimeError("missing_entailment_label")
            self._runtime = {
                "tokenizer": tokenizer,
                "session": session,
                "entailment_id": entailment_id,
                "contradiction_id": contradiction_id,
            }
        except Exception as exc:
            self._load_error = f"init_error:{type(exc).__name__}:{exc}"
            self._runtime = None
        return self._runtime

    def _has_local_model_bundle(self):
        return bool(self._resolve_model_path()) and os.path.isfile(os.path.join(self.model_dir, "config.json"))

    def _resolve_model_path(self):
        if not self.model_dir or not os.path.isdir(self.model_dir):
            return None
        candidates = [os.path.join(self.model_dir, filename) for filename in self.MODEL_FILENAMES]
        candidates.extend(os.path.join(self.model_dir, "onnx", filename) for filename in self.MODEL_FILENAMES)
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    def _config_label_map(self, config):
        label_map = {}
        for key, value in dict(getattr(config, "id2label", {}) or {}).items():
            try:
                label_map[int(key)] = str(value)
            except Exception:
                continue
        if label_map:
            return label_map
        for key, value in dict(getattr(config, "label2id", {}) or {}).items():
            try:
                label_map[int(value)] = str(key)
            except Exception:
                continue
        return label_map

    def _find_label_id(self, label_map, needle):
        needle = str(needle or "").strip().lower()
        for idx, label in (label_map or {}).items():
            if needle in str(label or "").strip().lower():
                return int(idx)
        return None

    def _normalize_scores(self, scores):
        total = float(sum(float(value) for value in (scores or {}).values()) or 0.0)
        if total <= 0.0:
            return {}
        return {key: float(value) / total for key, value in (scores or {}).items()}

    def _softmax(self, logits):
        arr = np.asarray(logits, dtype=np.float64)
        arr = arr - np.max(arr)
        exp = np.exp(arr)
        denom = float(np.sum(exp) or 1.0)
        return exp / denom

    def _safe_float(self, value, default=0.0):
        try:
            return float(value)
        except Exception:
            return default

    def _score_cache_key(self, premise, hypotheses):
        normalized_hypotheses = tuple(
            (str(label or "").strip(), str(hypothesis or "").strip())
            for label, hypothesis in (hypotheses or {}).items()
        )
        return (str(premise or "").strip(), normalized_hypotheses)


_INSTANCE = None


def get_element_relations_ai_enricher():
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ElementRelationsAIEnricher()
    return _INSTANCE
