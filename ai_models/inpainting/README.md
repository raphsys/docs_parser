# Inpainting CPU

Le pipeline de reconstruction peut maintenant restaurer le fond derrière le texte
via [background_inpainter.py](/home/raphael/Mes_Projets/docs_parser/background_inpainter.py).

Backends supportés:

- `lama_onnx`
  - open-source
  - CPU via `onnxruntime`
  - priorité si un modèle ONNX local est présent
- `opencv`
  - fallback CPU léger via `cv2.inpaint`

## Modèle recommandé

Le modèle recommandé est `lama_fp32.onnx` depuis `Carve/LaMa-ONNX` :

- licence `Apache-2.0`
- backend `lama_onnx`
- entrée fixe `512x512`
- source : `https://huggingface.co/Carve/LaMa-ONNX`

Le manifeste local est versionné dans `ai_models/inpainting/model_inventory.json`.

Téléchargement reproductible:

```bash
source .docs-parser/bin/activate
python3 scripts/download_inpainting_models.py
```

## Emplacement attendu des modèles

Par défaut, le code cherche un modèle LaMa ONNX dans:

```text
ai_models/inpainting/
  model_inventory.json
  lama/
    lama_fp32.onnx
```

Chemins alternatifs acceptés:

```text
ai_models/inpainting/lama/model.onnx
ai_models/inpainting/big-lama/model.onnx
ai_models/inpainting/lama.onnx
```

## Variables d'environnement

```bash
export BACKGROUND_INPAINT_ENABLE=1
export BACKGROUND_INPAINT_BACKEND=auto
export BACKGROUND_INPAINT_MODELS_ROOT=/chemin/vers/ai_models/inpainting
export BACKGROUND_INPAINT_RADIUS=3
```

Valeurs utiles pour `BACKGROUND_INPAINT_BACKEND`:

- `auto`
- `lama_onnx`
- `opencv`

## Comportement dans le pipeline

Le renderer utilise cette brique pour:

- les zones `text_erase_then_overlay`
- les `annotation_group` et autres groupes visuels traduits

Le fond est restauré sur un crop local, avec masque construit à partir des boîtes
de texte, puis le texte traduit est rerendu par-dessus.

En mode `auto`, le pipeline n'applique pas LaMa partout. Il bascule vers
`opencv` sur les masques trop denses ou trop fragmentés, car ce profil est plus
stable pour les documents annotés et évite des reconstructions trop
"hallucinées" sur les images sources.
