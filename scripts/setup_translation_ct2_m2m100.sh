#!/usr/bin/env bash
set -euo pipefail

# Setup local commercial-safe translation model:
# - source HF model: facebook/m2m100_418M (MIT)
# - runtime backend: CTranslate2 int8 on CPU
# - output dirs under ai_models/translation/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="${MODEL_ID:-facebook/m2m100_418M}"
OUT_CT2_DIR="${OUT_CT2_DIR:-$ROOT_DIR/ai_models/translation/m2m100_418m_ct2_int8}"
OUT_TOK_DIR="${OUT_TOK_DIR:-$ROOT_DIR/ai_models/translation/m2m100_418m_tokenizer}"
FORCE_FLAG="${FORCE_FLAG:---force}"

echo "[setup] Root: $ROOT_DIR"
echo "[setup] Model: $MODEL_ID"
echo "[setup] CT2 output: $OUT_CT2_DIR"
echo "[setup] Tokenizer output: $OUT_TOK_DIR"
echo "[setup] Force flag: $FORCE_FLAG"

python3 - <<'PY'
import importlib.util, sys
req = ["transformers", "sentencepiece", "ctranslate2"]
missing = [m for m in req if importlib.util.find_spec(m) is None]
if missing:
    print("[setup] Missing packages:", ", ".join(missing))
    print("[setup] Install with: pip install transformers sentencepiece ctranslate2")
    sys.exit(2)
print("[setup] Python dependencies OK")
PY

mkdir -p "$OUT_CT2_DIR" "$OUT_TOK_DIR"

python3 - <<PY
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("$MODEL_ID", use_fast=False)
tok.save_pretrained("$OUT_TOK_DIR")
print("[setup] Tokenizer saved:", "$OUT_TOK_DIR")
PY

ct2-transformers-converter \
  --model "$MODEL_ID" \
  --output_dir "$OUT_CT2_DIR" \
  --quantization int8 \
  --copy_files tokenizer_config.json special_tokens_map.json generation_config.json sentencepiece.bpe.model vocab.json \
  $FORCE_FLAG

# If conversion failed online but local snapshot exists, retry offline from local path.
if [ ! -f "$OUT_CT2_DIR/model.bin" ] && [ -d "$HOME/.cache/huggingface/hub/models--facebook--m2m100_418M/snapshots" ]; then
  SNAPSHOT_DIR="$(find "$HOME/.cache/huggingface/hub/models--facebook--m2m100_418M/snapshots" -maxdepth 1 -mindepth 1 -type d | head -n 1 || true)"
  if [ -n "$SNAPSHOT_DIR" ]; then
    echo "[setup] Retrying conversion from local snapshot: $SNAPSHOT_DIR"
    ct2-transformers-converter \
      --model "$SNAPSHOT_DIR" \
      --output_dir "$OUT_CT2_DIR" \
      --quantization int8 \
      --copy_files tokenizer_config.json special_tokens_map.json generation_config.json sentencepiece.bpe.model vocab.json \
      $FORCE_FLAG
  fi
fi

echo "[setup] Conversion done."
echo "[setup] Export these variables before starting server:"
echo "  export TRANSLATOR_BACKEND=ctranslate2"
echo "  export CT2_MODEL_DIR=\"$OUT_CT2_DIR\""
echo "  export CT2_TOKENIZER_DIR=\"$OUT_TOK_DIR\""
echo "  export TRANSLATOR_DEFAULT_SOURCE_LANG=en"
