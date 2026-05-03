"""
Tests des agents du pipeline — sans modèle LLM réel.

Tous les tests utilisent un ModelRuntime mock qui retourne des réponses
JSON prédéfinies, évitant le chargement des modèles (lourd en CI).
"""

from __future__ import annotations

import json
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_agents.base import ModelRuntime, ModelSpec, PipelineAgent, _extract_json
from pipeline_agents.p1_extraction import P1ExtractionAgent
from pipeline_agents.p2_structure import P2StructureAgent
from pipeline_agents.p3_layout import P3LayoutAgent
from pipeline_agents.p4_translation import P4TranslationAgent
from pipeline_agents.p5_render import P5RenderAgent
from pipeline_agents.p6_background import P6BackgroundAgent
from pipeline_agents.registry import AgentRegistry, get_agent, list_agents


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _MockRuntime(ModelRuntime):
    """Runtime factice — retourne une réponse JSON fixe."""

    def __init__(self, response_json: dict | None = None) -> None:
        spec = ModelSpec.__new__(ModelSpec)
        spec.requested = "mock"
        spec.resolved_id = "mock"
        spec.family = "generic"
        spec.backend = "mock"
        spec.ct2_dir = ""
        spec.trust_remote_code = False
        super().__init__(spec)
        self._mock_response = response_json or {}
        self._loaded = True
        self._pipe = {"backend": "mock"}

    def is_available(self) -> bool:
        return True

    def generate(self, messages: list[dict], max_new_tokens: int = 256) -> str:
        return json.dumps(self._mock_response, ensure_ascii=False)


class _UnavailableRuntime(ModelRuntime):
    """Runtime factice — toujours indisponible."""

    def __init__(self) -> None:
        spec = ModelSpec.__new__(ModelSpec)
        spec.requested = "unavailable"
        spec.resolved_id = "unavailable"
        spec.family = "generic"
        spec.backend = "mock"
        spec.ct2_dir = ""
        spec.trust_remote_code = False
        super().__init__(spec)
        self._loaded = True
        self._pipe = None

    def is_available(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Tests base.py
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_simple_object(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_embedded_in_text(self):
        assert _extract_json('some text {"key": "val"} more') == {"key": "val"}

    def test_nested(self):
        r = _extract_json('{"a":{"b":[1,2]}}')
        assert r == {"a": {"b": [1, 2]}}

    def test_no_json(self):
        assert _extract_json("no json here") is None

    def test_malformed(self):
        assert _extract_json("{broken") is None

    def test_array(self):
        assert _extract_json("[1,2,3]") == [1, 2, 3]

    def test_string_with_braces(self):
        r = _extract_json('{"msg": "use {braces} carefully"}')
        assert r == {"msg": "use {braces} carefully"}


class TestModelSpec:
    def test_alias_resolution(self):
        spec = ModelSpec("phi35")
        assert "phi35_mini_instruct" in spec.resolved_id

    def test_family_detection(self):
        spec = ModelSpec("phi35")
        assert spec.family == "phi"

    def test_unknown_alias_passthrough(self):
        spec = ModelSpec("/some/path/to/llama-model")
        assert spec.resolved_id == "/some/path/to/llama-model"

    def test_cache_key_unique_per_backend(self):
        s1 = ModelSpec("phi35", backend="transformers")
        s2 = ModelSpec("phi35", backend="ct2")
        assert s1.cache_key() != s2.cache_key()


class TestModelRuntime:
    def test_unavailable_returns_empty_string(self):
        rt = _UnavailableRuntime()
        assert rt.generate([{"role": "user", "content": "hi"}]) == ""

    def test_mock_runtime_generates(self):
        rt = _MockRuntime({"key": "value"})
        out = rt.generate([{"role": "user", "content": "test"}])
        assert '"key"' in out

    def test_global_cache_reuses_instance(self):
        rt1 = ModelRuntime.for_model("phi35-mini")
        rt2 = ModelRuntime.for_model("phi35-mini")
        assert rt1 is rt2


# ---------------------------------------------------------------------------
# Tests P1 ExtractionAgent
# ---------------------------------------------------------------------------

class TestP1ExtractionAgent:
    def _agent(self, response: dict) -> P1ExtractionAgent:
        return P1ExtractionAgent(_MockRuntime(response))

    def test_parse_heading(self):
        agent = self._agent({
            "c": [{"li": [0], "a": "heading"}, {"li": [1], "a": "keep"}],
            "sb": [], "lb": [], "lm": None, "hj": []
        })
        result = agent.run({"role": "body", "lines": [{"i": 0, "t": "Title"}, {"i": 1, "t": "Body"}]})
        assert result["c"][0]["a"] == "heading"
        assert result["c"][1]["a"] == "keep"

    def test_parse_atomic_caption(self):
        agent = self._agent({
            "c": [{"li": [0, 1, 2], "a": "atomic"}],
            "sb": [], "lb": [], "lm": None, "hj": []
        })
        result = agent.run({"role": "body", "lines": []})
        assert result["c"][0]["li"] == [0, 1, 2]

    def test_rejects_all_skip(self):
        agent = self._agent({
            "c": [{"li": [0], "a": "skip"}, {"li": [1], "a": "skip"}],
            "sb": [], "lb": [], "lm": None, "hj": []
        })
        result = agent.run({"role": "body", "lines": []})
        assert result["c"] == []

    def test_parse_hyphen_join(self):
        agent = self._agent({
            "c": [], "sb": [], "lb": [],
            "lm": None,
            "hj": [{"i": 0, "w": "guaranteed"}]
        })
        result = agent.run({"role": "body", "lines": []})
        assert result["hj"][0] == {"i": 0, "w": "guaranteed"}

    def test_parse_layout_mode(self):
        agent = self._agent({
            "c": [], "sb": [], "lb": [],
            "lm": {"line_flow": "inline_reflow", "breaks": [{"i": 0, "after": "soft_wrap"}]},
            "hj": []
        })
        result = agent.run({"role": "body", "lines": []})
        assert result["lm"]["line_flow"] == "inline_reflow"
        assert result["lm"]["breaks"][0]["after"] == "soft_wrap"

    def test_invalid_layout_mode_ignored(self):
        agent = self._agent({
            "c": [], "sb": [], "lb": [],
            "lm": {"line_flow": "INVALID_MODE"},
            "hj": []
        })
        result = agent.run({"role": "body", "lines": []})
        assert result["lm"] is None

    def test_parse_intra_line_split(self):
        agent = self._agent({
            "c": [], "sb": [{"i": 0, "s": ["Pooling reduces dimensions", "next stage applies normalization"]}],
            "lb": [], "lm": None, "hj": []
        })
        result = agent.run({"role": "body", "lines": []})
        assert len(result["sb"]) == 1
        assert len(result["sb"][0]["s"]) == 2

    def test_parse_inter_line_boundary(self):
        agent = self._agent({
            "c": [], "sb": [], "lb": [0, 2], "lm": None, "hj": []
        })
        result = agent.run({"role": "body", "lines": []})
        assert result["lb"] == [0, 2]

    def test_empty_response_returns_empty(self):
        agent = self._agent({})
        result = agent.run({"role": "body", "lines": []})
        assert result["c"] == []
        assert result["hj"] == []

    def test_unavailable_runtime_returns_empty(self):
        agent = P1ExtractionAgent(_UnavailableRuntime())
        result = agent.run({"role": "body", "lines": []})
        assert result == {}

    def test_invalid_action_ignored(self):
        agent = self._agent({
            "c": [{"li": [0], "a": "INVALID_ACTION"}],
            "sb": [], "lb": [], "lm": None, "hj": []
        })
        result = agent.run({"role": "body", "lines": []})
        assert result["c"] == []

    def test_score_block_hyphen(self):
        block = {"lines": [{"line_text": "guaran-"}, {"line_text": "teed to converge"}]}
        assert P1ExtractionAgent.score_block(block) >= 0.2

    def test_score_block_caption(self):
        block = {"lines": [{"line_text": "Figure 4.1 Overview"}]}
        assert P1ExtractionAgent.score_block(block) >= 0.25

    def test_score_block_no_signal(self):
        block = {"lines": [{"line_text": "This is a normal long paragraph with many words."}]}
        assert P1ExtractionAgent.score_block(block) == 0.0

    def test_build_messages_has_system_prompt(self):
        agent = self._agent({})
        msgs = agent.build_messages({"role": "body", "lines": []})
        assert msgs[0]["role"] == "system"
        assert "JSON" in msgs[0]["content"]

    def test_build_messages_includes_few_shot(self):
        agent = self._agent({})
        msgs = agent.build_messages({"role": "body", "lines": []})
        roles = [m["role"] for m in msgs]
        assert "user" in roles
        assert "assistant" in roles

    def test_cache_returns_same_result(self):
        response = {"c": [{"li": [0], "a": "heading"}], "sb": [], "lb": [], "lm": None, "hj": []}
        agent = P1ExtractionAgent(_MockRuntime(response))
        input_data = {"role": "body", "lines": [{"i": 0, "t": "Hello"}]}
        r1 = agent.run(input_data)
        r2 = agent.run(input_data)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Tests P2 StructureAgent
# ---------------------------------------------------------------------------

class TestP2StructureAgent:
    def _agent(self, response: dict) -> P2StructureAgent:
        return P2StructureAgent(_MockRuntime(response))

    def test_parse_groups(self):
        agent = self._agent({"groups": [["b0", "b1", "b2"]], "floating": [], "hierarchy": []})
        result = agent.run({"blocks": []})
        assert result["groups"] == [["b0", "b1", "b2"]]

    def test_parse_hierarchy(self):
        agent = self._agent({
            "groups": [], "floating": [],
            "hierarchy": [{"parent": "b0", "children": ["b1", "b2"]}]
        })
        result = agent.run({"blocks": []})
        assert result["hierarchy"][0]["parent"] == "b0"

    def test_parse_floating(self):
        agent = self._agent({"groups": [], "floating": ["b5", "b6"], "hierarchy": []})
        result = agent.run({"blocks": []})
        assert "b5" in result["floating"]

    def test_rejects_single_element_group(self):
        agent = self._agent({"groups": [["b0"]], "floating": [], "hierarchy": []})
        result = agent.run({"blocks": []})
        assert result["groups"] == []

    def test_empty_response(self):
        agent = self._agent({})
        result = agent.run({"blocks": []})
        assert result == {"groups": [], "floating": [], "hierarchy": []}


# ---------------------------------------------------------------------------
# Tests P3 LayoutAgent
# ---------------------------------------------------------------------------

class TestP3LayoutAgent:
    def _agent(self, response: dict) -> P3LayoutAgent:
        return P3LayoutAgent(_MockRuntime(response))

    def test_parse_inline_reflow(self):
        agent = self._agent({"layout_mode": "inline_reflow", "confidence": 0.92, "notes": "prose"})
        result = agent.run({})
        assert result["layout_mode"] == "inline_reflow"
        assert result["confidence"] == pytest.approx(0.92)

    def test_parse_preserve_line_breaks(self):
        agent = self._agent({"layout_mode": "preserve_line_breaks", "confidence": 0.85, "notes": "list"})
        result = agent.run({})
        assert result["layout_mode"] == "preserve_line_breaks"

    def test_invalid_mode_fallback(self):
        agent = self._agent({"layout_mode": "BOGUS", "confidence": 0.9})
        result = agent.run({})
        assert result["layout_mode"] == "inline_reflow"

    def test_confidence_clamped(self):
        agent = self._agent({"layout_mode": "fixed_lines", "confidence": 1.5})
        result = agent.run({})
        assert result["confidence"] == pytest.approx(1.0)

    def test_all_valid_modes(self):
        for mode in ["inline_reflow", "preserve_line_breaks", "preserve_paragraphs", "fixed_lines", "column_layout"]:
            agent = self._agent({"layout_mode": mode, "confidence": 0.8})
            result = agent.run({})
            assert result["layout_mode"] == mode

    def test_build_input_from_block(self):
        block = {
            "role": "body",
            "bbox": [10, 10, 200, 100],
            "lines": [
                {"line_text": "Step 1: Initialize the model", "bbox": [10, 10, 150, 20]},
                {"line_text": "Step 2: Train for 10 epochs", "bbox": [10, 25, 150, 35]},
            ],
        }
        inp = P3LayoutAgent.build_input_from_block(block)
        assert inp["role"] == "body"
        assert inp["line_count"] == 2
        assert len(inp["lines"]) == 2


# ---------------------------------------------------------------------------
# Tests P4 TranslationAgent
# ---------------------------------------------------------------------------

class TestP4TranslationAgent:
    def _agent(self, response: dict) -> P4TranslationAgent:
        return P4TranslationAgent(_MockRuntime(response))

    def test_parse_good_translation(self):
        agent = self._agent({"score": 0.95, "issues": [], "post_edit": None, "untranslated": []})
        result = agent.run({})
        assert result["score"] == pytest.approx(0.95)
        assert result["issues"] == []
        assert result["post_edit"] is None

    def test_parse_bad_translation_triggers_post_edit(self):
        agent = self._agent({
            "score": 0.4,
            "issues": [{"type": "omission", "desc": "not translated", "severity": "critical"}],
            "post_edit": "Texte corrigé.",
            "untranslated": ["original text"]
        })
        result = agent.run({})
        assert result["score"] == pytest.approx(0.4)
        assert result["post_edit"] == "Texte corrigé."
        assert "omission" in result["issues"][0]["type"]

    def test_post_edit_suppressed_if_score_high(self):
        agent = self._agent({
            "score": 0.75,
            "issues": [],
            "post_edit": "Some edit",
            "untranslated": []
        })
        result = agent.run({})
        assert result["post_edit"] is None  # score >= 0.7 → supprimé

    def test_invalid_issue_type_ignored(self):
        agent = self._agent({
            "score": 0.6,
            "issues": [{"type": "BOGUS_TYPE", "desc": "...", "severity": "major"}],
            "post_edit": None, "untranslated": []
        })
        result = agent.run({})
        assert result["issues"] == []

    def test_score_clamped(self):
        agent = self._agent({"score": 1.5, "issues": [], "post_edit": None, "untranslated": []})
        result = agent.run({})
        assert result["score"] == pytest.approx(1.0)

    def test_build_input_from_block(self):
        block = {
            "lines": [{"line_text": "The network learns"}, {"line_text": "hierarchical features."}],
            "translated_text": "Le réseau apprend des caractéristiques hiérarchiques.",
        }
        inp = P4TranslationAgent.build_input_from_block(block)
        assert inp["source_lang"] == "en"
        assert inp["target_lang"] == "fr"
        assert "réseau" in inp["translation"]

    def test_needs_validation_with_translation(self):
        block = {"translated_text": "Quelque chose."}
        assert P4TranslationAgent.needs_validation(block) is True

    def test_needs_validation_without_translation(self):
        block = {"lines": [{"line_text": "Hello"}]}
        assert P4TranslationAgent.needs_validation(block) is False


# ---------------------------------------------------------------------------
# Tests P5 RenderAgent
# ---------------------------------------------------------------------------

class TestP5RenderAgent:
    def _agent(self, response: dict) -> P5RenderAgent:
        return P5RenderAgent(_MockRuntime(response))

    def test_parse_prose_reflow(self):
        agent = self._agent({
            "strategy": "prose_reflow",
            "confidence": 0.92,
            "params": {"justify": True},
            "reason": "continuous prose"
        })
        result = agent.run({})
        assert result["strategy"] == "prose_reflow"
        assert result["params"]["justify"] is True

    def test_parse_label_stack(self):
        agent = self._agent({
            "strategy": "label_stack",
            "confidence": 0.88,
            "params": {"align": "left"},
            "reason": "short lines"
        })
        result = agent.run({})
        assert result["strategy"] == "label_stack"

    def test_invalid_strategy_fallback(self):
        agent = self._agent({"strategy": "BOGUS", "confidence": 0.9})
        result = agent.run({})
        assert result["strategy"] == "prose_reflow"

    def test_all_valid_strategies(self):
        for s in ["prose_reflow", "label_stack", "heading_reflow", "caption_reflow", "bitmap_preserve", "fixed_preserve"]:
            agent = self._agent({"strategy": s, "confidence": 0.9, "params": {}})
            result = agent.run({})
            assert result["strategy"] == s

    def test_confidence_clamped(self):
        agent = self._agent({"strategy": "prose_reflow", "confidence": -0.5})
        result = agent.run({})
        assert result["confidence"] == pytest.approx(0.0)

    def test_missing_params_defaults_to_empty_dict(self):
        agent = self._agent({"strategy": "prose_reflow", "confidence": 0.8})
        result = agent.run({})
        assert result["params"] == {}

    def test_build_input_from_block_body(self):
        block = {
            "role": "body",
            "bbox": [10, 10, 500, 100],
            "lines": [
                {"line_text": "Neural networks learn hierarchical feature representations."},
                {"line_text": "They use gradient descent to minimize the loss function."},
            ],
            "translated_text": "Les réseaux de neurones apprennent…",
        }
        inp = P5RenderAgent.build_input_from_block(block)
        assert inp["role"] == "body"
        assert inp["has_translation"] is True
        assert inp["line_count"] == 2

    def test_score_block_formula_returns_zero(self):
        block = {"role": "formula", "lines": []}
        assert P5RenderAgent.score_block(block) == 0.0

    def test_score_block_ambiguous_body(self):
        block = {
            "role": "body",
            "lines": [
                {"line_text": "Step 1"},
                {"line_text": "Step 2"},
                {"line_text": "Step 3"},
            ]
        }
        score = P5RenderAgent.score_block(block)
        assert score > 0.0


# ---------------------------------------------------------------------------
# Tests P6 BackgroundAgent
# ---------------------------------------------------------------------------

class TestP6BackgroundAgent:
    def _agent(self, response: dict) -> P6BackgroundAgent:
        return P6BackgroundAgent(_MockRuntime(response))

    def test_parse_clean_background(self):
        agent = self._agent({"quality": 0.95, "artifacts": [], "reprocess": [], "ok": True})
        result = agent.run({})
        assert result["quality"] == pytest.approx(0.95)
        assert result["ok"] is True

    def test_parse_dirty_background(self):
        agent = self._agent({
            "quality": 0.45,
            "artifacts": [{"region": [0, 0, 100, 50], "type": "text_residue", "severity": "high"}],
            "reprocess": [[0, 0, 100, 50]],
            "ok": False
        })
        result = agent.run({})
        assert result["ok"] is False
        assert result["artifacts"][0]["type"] == "text_residue"
        assert result["reprocess"][0] == [0.0, 0.0, 100.0, 50.0]

    def test_invalid_artifact_type_ignored(self):
        agent = self._agent({
            "quality": 0.7,
            "artifacts": [{"type": "BOGUS", "severity": "low"}],
            "reprocess": [], "ok": True
        })
        result = agent.run({})
        assert result["artifacts"] == []

    def test_ok_computed_from_quality_when_missing(self):
        agent = self._agent({
            "quality": 0.9,
            "artifacts": [],
            "reprocess": [],
            # ok absent → calculé automatiquement
        })
        result = agent.run({})
        assert result["ok"] is True

    def test_ok_false_if_high_severity(self):
        agent = self._agent({
            "quality": 0.88,
            "artifacts": [{"type": "shadow", "severity": "high"}],
            "reprocess": [],
        })
        result = agent.run({})
        assert result["ok"] is False

    def test_build_input_from_page(self):
        page = {
            "page_id": "p1",
            "blocks": [
                {"bbox": [0, 0, 100, 20], "inpainted": True, "inpaint_confidence": 0.9},
                {"bbox": [0, 25, 100, 45], "inpainted": False},
                {"bbox": [0, 50, 100, 70], "inpainted": True, "inpaint_confidence": 0.7},
            ]
        }
        inp = P6BackgroundAgent.build_input_from_page(page)
        assert inp["page_id"] == "p1"
        assert inp["blocks_removed"] == 2
        assert inp["coverage_ratio"] == pytest.approx(2 / 3, abs=1e-3)


# ---------------------------------------------------------------------------
# Tests du registre
# ---------------------------------------------------------------------------

class TestAgentRegistry:
    def setup_method(self):
        AgentRegistry.clear_instances()

    def test_list_agents_contains_all_stages(self):
        stages = list_agents()
        for s in ["p1_extraction", "p2_structure", "p3_layout", "p4_translation", "p5_render", "p6_background"]:
            assert s in stages

    def test_get_agent_returns_correct_type(self):
        # On ne charge pas le modèle — is_available() peut être False, c'est OK
        agent = get_agent("p5_render")
        assert isinstance(agent, P5RenderAgent)

    def test_get_agent_same_instance_cached(self):
        a1 = get_agent("p1_extraction")
        a2 = get_agent("p1_extraction")
        assert a1 is a2

    def test_get_agent_different_model_different_instance(self):
        a1 = get_agent("p5_render", model="phi35")
        a2 = get_agent("p5_render", model="qwen-small")
        assert a1 is not a2

    def test_get_agent_unknown_stage_raises(self):
        with pytest.raises(ValueError, match="non enregistré"):
            get_agent("p99_unknown")

    def test_custom_agent_registration(self):
        class MyAgent(P5RenderAgent):
            stage = "p5_render_custom"
        AgentRegistry.register("p5_render_custom", MyAgent)
        assert "p5_render_custom" in list_agents()
        AgentRegistry._classes.pop("p5_render_custom", None)

    def test_unavailable_agent_run_returns_empty(self):
        agent = P5RenderAgent(_UnavailableRuntime())
        result = agent.run({"role": "body"})
        assert result == {}


# ---------------------------------------------------------------------------
# Tests intégration P5RenderAgent ↔ DocumentReconstructor
# ---------------------------------------------------------------------------

class TestP5IntegrationWithReconstructor:
    """
    Teste l'étape G de compute_block_semantic_profile :
    affinage de la stratégie via P5RenderAgent (mock).
    """

    def setup_method(self):
        import importlib
        import reconstructor as rec_mod
        importlib.reload(rec_mod)
        self.rec_mod = rec_mod
        AgentRegistry.clear_instances()

    def _make_reconstructor(self):
        return self.rec_mod.DocumentReconstructor()

    def _block_ambiguous(self) -> dict:
        """Bloc body 3 lignes, 3 mots chacune → zone ambiguë prose/label."""
        return {
            "role": "body",
            "bbox": [10, 10, 150, 80],
            "lines": [
                {"line_text": "Step A here", "bbox": [10, 10, 140, 25]},
                {"line_text": "Step B there", "bbox": [10, 30, 140, 45]},
                {"line_text": "Step C done", "bbox": [10, 50, 140, 65]},
            ],
        }

    def _inject_mock_agent(self, reconstructor, ai_strategy: str, confidence: float = 0.85):
        """Injecte un agent mock dans l'instance de reconstructeur."""
        agent = P5RenderAgent(_MockRuntime({
            "strategy": ai_strategy,
            "confidence": confidence,
            "params": {},
            "reason": "test",
        }))
        reconstructor._render_agent = agent
        reconstructor._render_agent_loaded = True

    def test_agent_disabled_by_default(self):
        """Sans PIPELINE_AGENT_RENDER_ENABLE=1, l'agent ne doit pas être chargé."""
        env_before = os.environ.get("PIPELINE_AGENT_RENDER_ENABLE")
        os.environ.pop("PIPELINE_AGENT_RENDER_ENABLE", None)
        try:
            rec = self._make_reconstructor()
            assert rec._get_render_agent() is None
        finally:
            if env_before is not None:
                os.environ["PIPELINE_AGENT_RENDER_ENABLE"] = env_before

    def test_heuristic_preserved_when_agent_disabled(self):
        """Le résultat heuristique ne doit pas changer quand l'agent est absent."""
        rec = self._make_reconstructor()
        # Pas d'agent injecté → _render_agent reste None
        rec._render_agent_loaded = True  # évite la tentative de chargement
        block = self._block_ambiguous()
        strategy = rec._ai_refine_render_strategy(block, "prose_reflow")
        assert strategy == "prose_reflow"

    def test_agent_overrides_prose_to_label_with_high_confidence(self):
        """Quand l'agent est confiant (≥ 0.70) et le bloc est ambigu, override."""
        rec = self._make_reconstructor()
        self._inject_mock_agent(rec, "label_stack", confidence=0.88)
        block = self._block_ambiguous()
        strategy = rec._ai_refine_render_strategy(block, "prose_reflow")
        assert strategy == "label_stack"

    def test_agent_overrides_label_to_prose_with_high_confidence(self):
        """Override inverse : label_stack → prose_reflow."""
        rec = self._make_reconstructor()
        self._inject_mock_agent(rec, "prose_reflow", confidence=0.91)
        block = {
            "role": "body",
            "bbox": [10, 10, 150, 80],
            "lines": [
                {"line_text": "val A", "bbox": [10, 10, 80, 25]},
                {"line_text": "val B", "bbox": [10, 30, 80, 45]},
                {"line_text": "val C", "bbox": [10, 50, 80, 65]},
            ],
        }
        strategy = rec._ai_refine_render_strategy(block, "label_stack")
        assert strategy == "prose_reflow"

    def test_low_confidence_keeps_heuristic(self):
        """Confiance < 0.70 → pas d'override."""
        rec = self._make_reconstructor()
        self._inject_mock_agent(rec, "label_stack", confidence=0.60)
        block = self._block_ambiguous()
        strategy = rec._ai_refine_render_strategy(block, "prose_reflow")
        assert strategy == "prose_reflow"

    def test_non_ambiguous_block_skips_agent(self):
        """Bloc non ambigu (score_block < 0.4) → pas d'appel agent."""
        rec = self._make_reconstructor()
        call_count = {"n": 0}

        class CountingAgent(P5RenderAgent):
            def run(self, input_data, **kw):
                call_count["n"] += 1
                return {"strategy": "label_stack", "confidence": 0.9, "params": {}, "reason": ""}

        rec._render_agent = CountingAgent(_MockRuntime({}))
        rec._render_agent_loaded = True

        # Bloc très long → score faible
        block = {
            "role": "body",
            "bbox": [10, 10, 600, 100],
            "lines": [
                {"line_text": "This is a very long prose paragraph with many words on this line."},
                {"line_text": "Another equally long sentence that clearly belongs to body prose."},
                {"line_text": "A third line confirming this is continuous flowing text content."},
            ],
        }
        rec._ai_refine_render_strategy(block, "prose_reflow")
        assert call_count["n"] == 0

    def test_agent_cannot_override_non_ambiguous_strategies(self):
        """L'étape G ne peut changer que prose_reflow ↔ label_stack, pas heading/bitmap."""
        rec = self._make_reconstructor()
        self._inject_mock_agent(rec, "prose_reflow", confidence=0.95)
        # heading_reflow n'est pas dans la zone ambiguë → pas de changement
        block = self._block_ambiguous()
        strategy = rec._ai_refine_render_strategy(block, "heading_reflow")
        assert strategy == "heading_reflow"

    def test_compute_block_semantic_profile_ai_refinement_applied(self):
        """compute_block_semantic_profile intègre l'override IA sur bloc ambigu."""
        rec = self._make_reconstructor()
        self._inject_mock_agent(rec, "label_stack", confidence=0.85)
        block = self._block_ambiguous()
        profile = rec.compute_block_semantic_profile(block, page_data=None)
        assert profile is not None
        assert profile.render_strategy == "label_stack"
        assert profile.content_class == "label"

    def test_compute_block_semantic_profile_no_ai_when_disabled(self):
        """Sans agent, compute_block_semantic_profile retourne la stratégie heuristique."""
        rec = self._make_reconstructor()
        rec._render_agent_loaded = True  # simulate: not enabled
        block = {
            "role": "body",
            "bbox": [10, 10, 600, 200],
            "lines": [
                {"line_text": "Neural networks learn hierarchical feature representations from data."},
                {"line_text": "They use gradient descent to minimize the cross-entropy loss function."},
                {"line_text": "The optimizer updates weights via backpropagation through all layers."},
            ],
        }
        profile = rec.compute_block_semantic_profile(block, page_data=None)
        assert profile is not None
        assert profile.render_strategy == "prose_reflow"


# ---------------------------------------------------------------------------
# Tests intégration P1ExtractionAgent ↔ ocr_server._postprocess_blocks_semantic
# ---------------------------------------------------------------------------

class TestP1IntegrationWithOcrServer:
    """
    Teste _postprocess_blocks_semantic et _p1_agent_postprocess_blocks
    dans ocr_server.py via mock agent.
    """

    def setup_method(self):
        import importlib
        import ocr_server as ocr_mod
        importlib.reload(ocr_mod)
        self.ocr = ocr_mod
        AgentRegistry.clear_instances()

    def _simple_block(self, role: str = "body", lines: list | None = None) -> dict:
        if lines is None:
            lines = [
                {"line_index": 0, "line_text": "The algorithm is guaran-", "bbox": [10, 10, 200, 25]},
                {"line_index": 1, "line_text": "teed to converge eventually.", "bbox": [10, 30, 200, 45]},
            ]
        return {
            "id": "b0",
            "role": role,
            "bbox": [10, 10, 200, 50],
            "lines": lines,
            "semantic_phrases": [{"text": "The algorithm is guaranteed to converge eventually.", "sentence_end_reason": "eof"}],
        }

    def _inject_p1_agent(self, ai_response: dict):
        """Injecte un agent P1 mock dans le registre."""
        agent = P1ExtractionAgent(_MockRuntime(ai_response))
        AgentRegistry._instances["p1_extraction|phi35|auto"] = agent
        return agent

    def test_dispatcher_uses_llm_corrector_by_default(self, monkeypatch):
        """Sans PIPELINE_AGENT_P1_ENABLE, le dispatcher appelle _llm_postprocess_blocks."""
        called = {"llm": 0, "p1": 0}
        monkeypatch.setattr(self.ocr, "_llm_postprocess_blocks", lambda b: called.__setitem__("llm", called["llm"] + 1))
        monkeypatch.setattr(self.ocr, "_p1_agent_postprocess_blocks", lambda b: called.__setitem__("p1", called["p1"] + 1))
        monkeypatch.delenv("PIPELINE_AGENT_P1_ENABLE", raising=False)
        self.ocr._postprocess_blocks_semantic([])
        assert called["llm"] == 1
        assert called["p1"] == 0

    def test_dispatcher_uses_p1_agent_when_enabled(self, monkeypatch):
        """Avec PIPELINE_AGENT_P1_ENABLE=1, le dispatcher appelle _p1_agent_postprocess_blocks."""
        called = {"llm": 0, "p1": 0}
        monkeypatch.setattr(self.ocr, "_llm_postprocess_blocks", lambda b: called.__setitem__("llm", called["llm"] + 1))
        monkeypatch.setattr(self.ocr, "_p1_agent_postprocess_blocks", lambda b: called.__setitem__("p1", called["p1"] + 1))
        monkeypatch.setenv("PIPELINE_AGENT_P1_ENABLE", "1")
        self.ocr._postprocess_blocks_semantic([])
        assert called["llm"] == 0
        assert called["p1"] == 1

    def test_p1_agent_applies_heading_correction(self, monkeypatch):
        """Un agent P1 qui retourne heading doit modifier semantic_phrases du bloc."""
        monkeypatch.setenv("PIPELINE_AGENT_P1_ENABLE", "1")
        monkeypatch.setenv("PIPELINE_AGENT_P1_THRESHOLD", "0.0")  # seuil=0 → tous les blocs
        self._inject_p1_agent({
            "c": [{"li": [0], "a": "heading"}],
            "sb": [], "lb": [], "lm": None, "hj": []
        })
        block = self._simple_block()
        # Pré-condition : semantic_phrases existantes sans heading_line
        assert not any(
            sp.get("sentence_end_reason") == "heading_line"
            for sp in block.get("semantic_phrases", [])
        )
        self.ocr._p1_agent_postprocess_blocks([block])
        reasons = {sp.get("sentence_end_reason") for sp in block.get("semantic_phrases", [])}
        assert "heading_line" in reasons

    def test_p1_agent_applies_hyphen_join(self, monkeypatch):
        """Un agent P1 qui retourne hj doit corriger le tiret de césure dans les lignes."""
        monkeypatch.setenv("PIPELINE_AGENT_P1_THRESHOLD", "0.0")
        self._inject_p1_agent({
            "c": [], "sb": [], "lb": [], "lm": None,
            "hj": [{"i": 0, "w": "guaranteed"}]
        })
        block = self._simple_block()
        self.ocr._p1_agent_postprocess_blocks([block])
        # La ligne 0 doit avoir le mot complet dans line_text (guaran- → guaranteed)
        line_0 = next(
            ln for ln in block.get("lines", [])
            if int(ln.get("line_index", -1)) == 0
        )
        assert "guaranteed" in line_0.get("line_text", "")
        assert "guaran-" not in line_0.get("line_text", "")

    def test_p1_agent_skips_block_below_threshold(self, monkeypatch):
        """Un bloc dont le score est sous le seuil ne doit pas être traité."""
        monkeypatch.setenv("PIPELINE_AGENT_P1_THRESHOLD", "0.99")  # seuil très élevé
        call_count = {"n": 0}

        class CountingAgent(P1ExtractionAgent):
            def run(self, input_data, **kw):
                call_count["n"] += 1
                return {}

        AgentRegistry._instances["p1_extraction|phi35|auto"] = CountingAgent(_MockRuntime({}))
        block = self._simple_block()
        # bloc normal sans signal fort → score faible → doit être skipé
        block["lines"] = [{"line_index": 0, "line_text": "This is normal long text prose.", "bbox": [10, 10, 200, 25]}]
        block["semantic_phrases"] = [{"text": "This is normal long text prose.", "sentence_end_reason": "eof"}]
        self.ocr._p1_agent_postprocess_blocks([block])
        assert call_count["n"] == 0

    def test_p1_agent_rejects_regression(self, monkeypatch):
        """Si les corrections régressent la qualité, les phrases heuristiques sont restaurées."""
        monkeypatch.setenv("PIPELINE_AGENT_P1_THRESHOLD", "0.0")
        # Agent qui marque toutes les lignes comme "skip" → résultat vide → régression
        self._inject_p1_agent({
            "c": [],  # rejeté par parse (all-skip) → pas de corrections
            "sb": [], "lb": [], "lm": None, "hj": []
        })
        block = self._simple_block()
        original_phrases = list(block.get("semantic_phrases", []))
        self.ocr._p1_agent_postprocess_blocks([block])
        # Sans corrections, les phrases originales doivent être conservées
        assert block.get("semantic_phrases") == original_phrases

    def test_p1_agent_unavailable_is_silent(self, monkeypatch):
        """Si l'agent est indisponible, _p1_agent_postprocess_blocks ne lève pas."""
        AgentRegistry._instances["p1_extraction|phi35|auto"] = P1ExtractionAgent(_UnavailableRuntime())
        block = self._simple_block()
        # Ne doit pas lever d'exception
        self.ocr._p1_agent_postprocess_blocks([block])
