#!/usr/bin/env bash
set -euo pipefail

# Setup local high-quality EN->FR translation model:
# - preferred source HF model: Helsinki-NLP/opus-mt-tc-big-en-fr (Apache-2.0)
# - offline fallback: cached snapshot of Helsinki-NLP/opus-mt-en-fr
# - runtime backend: CTranslate2 int8 on CPU
# - output dirs under ai_models/translation/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="${MODEL_ID:-Helsinki-NLP/opus-mt-tc-big-en-fr}"
OUT_CT2_DIR="${OUT_CT2_DIR:-$ROOT_DIR/ai_models/translation/opus_mt_tc_big_en_fr_ct2_int8}"
OUT_TOK_DIR="${OUT_TOK_DIR:-$ROOT_DIR/ai_models/translation/opus_mt_tc_big_en_fr_tokenizer}"
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

MODEL_SRC="$MODEL_ID"
TRY_ONLINE="${TRY_ONLINE:-0}"
TC_BIG_CACHE="$(find "$HOME/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-tc-big-en-fr/snapshots" -maxdepth 1 -mindepth 1 -type d -exec test -f '{}/config.json' ';' -print 2>/dev/null | head -n 1 || true)"
OPUS_CACHE="$(find "$HOME/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-en-fr/snapshots" -maxdepth 1 -mindepth 1 -type d -exec test -f '{}/config.json' ';' -print 2>/dev/null | head -n 1 || true)"

ONLINE_OK=1
if [ "$TRY_ONLINE" != "1" ]; then
  ONLINE_OK=0
fi
if [ "$ONLINE_OK" = "1" ] && ! python3 - <<PY
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("$MODEL_ID", use_fast=False)
print("ok")
PY
then
  ONLINE_OK=0
fi

if [ "$ONLINE_OK" != "1" ]; then
  if [ -n "$TC_BIG_CACHE" ]; then
    MODEL_SRC="$TC_BIG_CACHE"
    echo "[setup] Using cached tc-big snapshot: $MODEL_SRC"
  elif [ -n "$OPUS_CACHE" ]; then
    MODEL_SRC="$OPUS_CACHE"
    echo "[setup] Using cached opus-mt-en-fr snapshot: $MODEL_SRC"
  else
    echo "[setup] ERROR: neither online model nor local cache snapshot available."
    exit 1
  fi
fi

python3 - <<PY
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("$MODEL_SRC", use_fast=False, local_files_only=True if "$ONLINE_OK" != "1" else False)
tok.save_pretrained("$OUT_TOK_DIR")
print("[setup] Tokenizer saved:", "$OUT_TOK_DIR")
PY

ct2-transformers-converter \
  --model "$MODEL_SRC" \
  --output_dir "$OUT_CT2_DIR" \
  --quantization int8 \
  --copy_files tokenizer_config.json generation_config.json source.spm target.spm vocab.json \
  $FORCE_FLAG

echo "[setup] Conversion done."
echo "[setup] Export these variables before starting server:"
echo "  export TRANSLATOR_BACKEND=ctranslate2"
echo "  export TRANSLATOR_MODEL_FAMILY=marian"
echo "  export CT2_MODEL_DIR=\"$OUT_CT2_DIR\""
echo "  export CT2_TOKENIZER_DIR=\"$OUT_TOK_DIR\""
echo "  export TRANSLATOR_DEFAULT_SOURCE_LANG=en"
