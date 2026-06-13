"""Shared fakes for CTranslate2 engine tests.

These let us exercise the real encode/decode/batch logic of CTranslate2Engine
without the heavy models, by mocking ``ctranslate2.Translator`` and
``transformers.AutoTokenizer``.
"""

from __future__ import annotations

import json
from pathlib import Path


_SPECIAL = {"</s>", "<s>", "<pad>", "<unk>"}


class FakeTokenizer:
    """Minimal SentencePiece-like tokenizer with a growing vocabulary."""

    def __init__(self, family: str = "marian"):
        self.family = family
        self.src_lang = None
        self._id2tok: dict[int, str] = {}
        self._tok2id: dict[str, int] = {}
        # M2M100/NLLB language tokens.
        self.lang_code_to_token = {"en": "__en__", "fr": "__fr__", "es": "__es__"}

    def _id_for(self, token: str) -> int:
        if token not in self._tok2id:
            new_id = len(self._tok2id)
            self._tok2id[token] = new_id
            self._id2tok[new_id] = token
        return self._tok2id[token]

    def encode(self, text: str):
        tokens = [f"▁{word}" for word in str(text).split()] + ["</s>"]
        return [self._id_for(tok) for tok in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self._id2tok.get(i, "<unk>") for i in ids]

    def convert_tokens_to_ids(self, tokens):
        return [self._id_for(str(tok)) for tok in tokens]

    def decode(self, ids, skip_special_tokens: bool = True):
        tokens = [self._id2tok.get(i, "") for i in ids]
        if skip_special_tokens:
            tokens = [tok for tok in tokens if tok not in _SPECIAL and not tok.startswith("__")]
        text = "".join(tokens).replace("▁", " ")
        return text.strip()


class FakeResult:
    def __init__(self, hypotheses):
        self.hypotheses = hypotheses


class FakeTranslator:
    """Records translate_batch calls and returns deterministic hypotheses."""

    def __init__(self, *args, hypothesis_builder=None, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.calls: list[dict] = []
        self._hypothesis_builder = hypothesis_builder

    def translate_batch(self, batch_tokens, target_prefix=None, max_batch_size=None, **kwargs):
        self.calls.append({
            "batch_tokens": [list(toks) for toks in batch_tokens],
            "target_prefix": target_prefix,
            "max_batch_size": max_batch_size,
        })
        results = []
        for index, toks in enumerate(batch_tokens):
            if self._hypothesis_builder is not None:
                hyp = self._hypothesis_builder(index, list(toks), target_prefix)
            else:
                # Default: echo source words as the "translation".
                words = [tok for tok in toks if tok not in _SPECIAL]
                hyp = words or ["▁ok"]
            results.append(FakeResult(hypotheses=[hyp]))
        return results


def write_inventory(tmp_path: Path, models: list[dict]) -> Path:
    """Write a flat ('models') inventory and create the referenced dirs."""
    entries = []
    for model in models:
        model_dir = tmp_path / model["name"] / "ct2"
        tokenizer_dir = tmp_path / model["name"] / "tok"
        if model.get("available", True):
            model_dir.mkdir(parents=True, exist_ok=True)
            tokenizer_dir.mkdir(parents=True, exist_ok=True)
        entries.append({
            "name": model["name"],
            "family": model.get("family", "generic"),
            "model_dir": str(model_dir),
            "tokenizer_dir": str(tokenizer_dir),
            "source_langs": model.get("source_langs", []),
            "target_langs": model.get("target_langs", []),
        })
    inventory = tmp_path / "model_inventory.json"
    inventory.write_text(json.dumps({"default_engine": "ct2", "models": entries}), encoding="utf-8")
    return inventory


def install_fakes(monkeypatch, tokenizer: FakeTokenizer, translator: FakeTranslator):
    """Patch ctranslate2.Translator and transformers.AutoTokenizer."""
    import ctranslate2
    import transformers

    monkeypatch.setattr(ctranslate2, "Translator", lambda *a, **k: translator)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **k: tokenizer))
