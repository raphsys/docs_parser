Model root reserve a `ElementRelationsAIEnricher`.

Objectif:

- desambiguaiser les relations ambiguës entre phrases consecutives dans un bloc
- confirmer `continuation` vs `new_unit`
- ajuster `logical_relation` quand les heuristiques restent incertaines

Backend attendu:

- `onnx_nli`

Modele recommande pour CPU:

- `cross-encoder/nli-deberta-v3-xsmall`

Pourquoi:

- bon compromis qualite / latence sur CPU
- licence `Apache-2.0`
- le repo contient deja les variantes ONNX quantifiees CPU

Repo recommande:

- `https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall`

Arborescence attendue:

- `config.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `tokenizer.json` ou les fichiers tokenizer natifs du modele
- `spm.model` si present dans le repo source
- `onnx/model_quint8_avx2.onnx` pour CPU x86 AVX2

Fichiers ONNX acceptes par le loader:

- `model.onnx`
- `model_quint8_avx2.onnx`
- `model_qint8_arm64.onnx`
- `model_qint8_avx512.onnx`
- `model_qint8_avx512_vnni.onnx`
- `model_O1.onnx`
- `model_O2.onnx`
- `model_O3.onnx`
- `model_O4.onnx`

Le loader cherche ces noms:

- a la racine du dossier
- puis dans `onnx/`

Exemple de structure minimale:

```text
ai_models/element_relations_nli/
  config.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  onnx/
    model_quint8_avx2.onnx
```

Variables d'environnement:

- `ELEMENT_RELATIONS_AI_ENABLE=1`
- `ELEMENT_RELATIONS_AI_MODEL_DIR=/chemin/vers/ai_models/element_relations_nli`
- `ELEMENT_RELATIONS_AI_BACKEND=onnx_nli`
- `ELEMENT_RELATIONS_AI_MIN_CONFIDENCE=0.78`

Comportement:

- si le bundle local est present, l'enricher peut s'activer automatiquement
- sinon, le pipeline reste sur les heuristiques et expose un statut non applique

Contrat de sortie:

- `page_data["element_relations_ai"]`
- `page_data["layout"]["element_relations_ai"]`
- `relation["heuristic_decision"]`
- `relation["semantic_ai_review"]`
- `relation["resolved_by"]`
