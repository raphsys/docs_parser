#!/usr/bin/env bash
set -euo pipefail

# Verbose end-to-end pipeline:
# 1) download datasets
# 2) generate synthetic dataset
# 3) train embedder
# 4) export ONNX
# 5) build font index
# 6) smoke test matcher + OCR integration

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./.docs-parser/bin/python}"
LOG_DIR="${LOG_DIR:-./logs/font_ai_pipeline}"
DATA_SOURCES="${DATA_SOURCES:-google_fonts_repo chars74k_english_fnt}"
DOWNLOAD_OUTPUT_DIR="${DOWNLOAD_OUTPUT_DIR:-./datasets/font_sources}"
SYNTH_DATASET_DIR="${SYNTH_DATASET_DIR:-./datasets/font_ai_synth}"
MODEL_DIR="${MODEL_DIR:-./ai_models/fonts}"
MAX_FONTS="${MAX_FONTS:-1200}"
SAMPLES_PER_FONT="${SAMPLES_PER_FONT:-300}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
WORKERS="${WORKERS:-4}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
DOWNLOAD_ENABLED="${DOWNLOAD_ENABLED:-1}"
DEVICE="${DEVICE:-cpu}"
SCHEDULER="${SCHEDULER:-cosine}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-4}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.001}"
AUG_STRENGTH="${AUG_STRENGTH:-0.35}"
NO_AUGMENT="${NO_AUGMENT:-0}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"
STORIA_ONLY="${STORIA_ONLY:-1}"
STORIA_MODEL_PATH="${STORIA_MODEL_PATH:-./ai_models/fonts/teacher/model.onnx}"
MODEL_VARIANT="${MODEL_VARIANT:-small}"
HNM_ENABLED="${HNM_ENABLED:-1}"
HNM_TOP_K="${HNM_TOP_K:-20}"
HNM_REFRESH_EVERY="${HNM_REFRESH_EVERY:-1}"
HNM_WEIGHT="${HNM_WEIGHT:-0.08}"
HNM_MARGIN="${HNM_MARGIN:-0.35}"
TEACHER_ONNX_PATH="${TEACHER_ONNX_PATH:-}"
TEACHER_WEIGHT="${TEACHER_WEIGHT:-0.15}"
TEACHER_SIM_TEMP="${TEACHER_SIM_TEMP:-0.12}"
TEACHER_MIX="${TEACHER_MIX:-0.35}"
TEACHER_STUDENT_TEMP="${TEACHER_STUDENT_TEMP:-2.0}"
TEACHER_WARMUP_EPOCHS="${TEACHER_WARMUP_EPOCHS:-0}"
TRAIN_EXTRA_ARGS=()
if [[ "$NO_AUGMENT" == "1" ]]; then
  TRAIN_EXTRA_ARGS+=("--no-augment")
fi
if [[ -n "$TEACHER_ONNX_PATH" ]]; then
  TRAIN_EXTRA_ARGS+=("--teacher-onnx-path" "$TEACHER_ONNX_PATH")
fi

mkdir -p "$LOG_DIR"
mkdir -p "$MODEL_DIR"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }
TOTAL_STEPS=7
CURRENT_STEP=0

run_step() {
  local name="$1"
  shift
  local logfile="$LOG_DIR/${name}.log"
  CURRENT_STEP=$((CURRENT_STEP + 1))
  log "START: [$CURRENT_STEP/$TOTAL_STEPS] $name"
  log "CMD: $*"
  PYTHONUNBUFFERED=1 "$@" 2>&1 | tee "$logfile"
  log "DONE:  [$CURRENT_STEP/$TOTAL_STEPS] $name (log: $logfile)"
}

log "Pipeline root: $ROOT_DIR"
log "Python: $PYTHON_BIN"
log "Logs: $LOG_DIR"

if [[ "$STORIA_ONLY" == "1" ]]; then
  CURRENT_STEP=$((CURRENT_STEP + 1))
  log "SKIP:  [$CURRENT_STEP/$TOTAL_STEPS] Storia-only mode: skipping download"
elif [[ "$TRAIN_ONLY" == "1" ]]; then
  CURRENT_STEP=$((CURRENT_STEP + 1))
  log "SKIP:  [$CURRENT_STEP/$TOTAL_STEPS] train-only mode: skipping download"
elif [[ "$DOWNLOAD_ENABLED" == "1" ]]; then
  run_step "01_download" \
    "$PYTHON_BIN" download_font_datasets.py \
      --source $DATA_SOURCES \
      --output-dir "$DOWNLOAD_OUTPUT_DIR" \
      --extract
else
  CURRENT_STEP=$((CURRENT_STEP + 1))
  log "SKIP:  [$CURRENT_STEP/$TOTAL_STEPS] download step disabled (DOWNLOAD_ENABLED=$DOWNLOAD_ENABLED)"
fi

if [[ "$STORIA_ONLY" == "1" ]]; then
  CURRENT_STEP=$((CURRENT_STEP + 1))
  log "SKIP:  [$CURRENT_STEP/$TOTAL_STEPS] Storia-only mode: skipping synth generation"
elif [[ "$TRAIN_ONLY" == "1" ]]; then
  CURRENT_STEP=$((CURRENT_STEP + 1))
  log "SKIP:  [$CURRENT_STEP/$TOTAL_STEPS] train-only mode: skipping synth generation"
  if [[ ! -f "$SYNTH_DATASET_DIR/metadata.csv" ]]; then
    log "ERROR: metadata not found at $SYNTH_DATASET_DIR/metadata.csv"
    log "Hint: run once without TRAIN_ONLY or set SYNTH_DATASET_DIR to an existing dataset."
    exit 1
  fi
else
  run_step "02_generate_synth" \
    "$PYTHON_BIN" generate_synthetic_font_dataset.py \
      --output-dir "$SYNTH_DATASET_DIR" \
      --max-fonts "$MAX_FONTS" \
      --samples-per-font "$SAMPLES_PER_FONT" \
      --image-size "$IMAGE_SIZE"
fi

if [[ "$STORIA_ONLY" == "1" ]]; then
  CURRENT_STEP=$((CURRENT_STEP + 1))
  log "SKIP:  [$CURRENT_STEP/$TOTAL_STEPS] Storia-only mode: skipping training"
else
  run_step "03_train" \
    "$PYTHON_BIN" train_font_embedder.py \
      --dataset-dir "$SYNTH_DATASET_DIR" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --workers "$WORKERS" \
      --embedding-dim "$EMBEDDING_DIM" \
      --image-size "$IMAGE_SIZE" \
      --out-dir "$MODEL_DIR" \
      --device "$DEVICE" \
      --export-onnx \
      --scheduler "$SCHEDULER" \
      --early-stop-patience "$EARLY_STOP_PATIENCE" \
      --early-stop-min-delta "$EARLY_STOP_MIN_DELTA" \
      --model-variant "$MODEL_VARIANT" \
      --aug-strength "$AUG_STRENGTH" \
      --hnm-enabled "$HNM_ENABLED" \
      --hnm-top-k "$HNM_TOP_K" \
      --hnm-refresh-every "$HNM_REFRESH_EVERY" \
      --hnm-weight "$HNM_WEIGHT" \
      --hnm-margin "$HNM_MARGIN" \
      --teacher-weight "$TEACHER_WEIGHT" \
      --teacher-sim-temp "$TEACHER_SIM_TEMP" \
      --teacher-mix "$TEACHER_MIX" \
      --teacher-student-temp "$TEACHER_STUDENT_TEMP" \
      --teacher-warmup-epochs "$TEACHER_WARMUP_EPOCHS" \
      "${TRAIN_EXTRA_ARGS[@]}"
fi

run_step "04_build_index" \
  "$PYTHON_BIN" build_font_index.py \
    --model "$STORIA_MODEL_PATH" \
    --index "$MODEL_DIR/font_embedding_index.npz" \
    --max-fonts "$MAX_FONTS"

run_step "05_quality_report" \
  env FONT_AI_MODEL_DIR="$MODEL_DIR" STORIA_ONLY="$STORIA_ONLY" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

p = Path(os.getenv("FONT_AI_MODEL_DIR", "./ai_models/fonts")) / "train_report.json"
if os.getenv("STORIA_ONLY", "1") == "1":
    print("mode=storia_only")
    print("training=disabled")
    print("teacher_model=./ai_models/fonts/teacher/model.onnx")
elif not p.exists():
    print("report_missing=", p)
else:
    r = json.loads(p.read_text(encoding="utf-8"))
    retr = r.get("retrieval", {})
    print("best_epoch=", r.get("best_epoch"))
    print("scheduler=", r.get("scheduler"))
    print("aug_strength=", r.get("aug_strength"))
    print("model_variant=", r.get("model_variant"))
    print("classifier_acc1=", r.get("best_val_acc1_classifier"))
    print("classifier_acc5=", r.get("best_val_acc5_classifier"))
    print("retrieval_top1=", retr.get("top1_acc"))
    print("retrieval_top5=", retr.get("top5_acc"))
    print("recommended_threshold_balanced=", retr.get("recommended_threshold_balanced"))
    print("recommended_threshold_high_precision=", retr.get("recommended_threshold_high_precision"))
    print("hard_negative_pairs=", len(retr.get("hard_negative_pairs", [])))
PY

run_step "06_smoke_matcher" \
  env FONT_AI_MODEL_PATH="$STORIA_MODEL_PATH" FONT_AI_INDEX_PATH="$MODEL_DIR/font_embedding_index.npz" "$PYTHON_BIN" - <<'PY'
from font_ai_matcher import FontAIMatcher
m = FontAIMatcher()
print("matcher_ready=", m.is_ready())
if m.is_ready():
    print("storia_classes=", len(m._index_names))
PY

run_step "07_smoke_ocr_integration" \
  env FONT_AI_MODEL_PATH="$STORIA_MODEL_PATH" FONT_AI_INDEX_PATH="$MODEL_DIR/font_embedding_index.npz" "$PYTHON_BIN" - <<'PY'
import asyncio
import json
from starlette.datastructures import UploadFile
from ocr_server import perform_ocr

async def main():
    with open("tests/test_doc.jpg", "rb") as f:
        up = UploadFile(filename="test_doc.jpg", file=f)
        resp = await perform_ocr(file=up, force_ai=True)
        payload = json.loads(resp.body.decode("utf-8"))
        print("status=", payload.get("status"))
        first = payload["results"][0]
        spans = []
        for b in first["structure"]["blocks"]:
            for l in b.get("lines", []):
                for p in l.get("phrases", []):
                    spans.extend(p.get("spans", []))
        tagged = [s for s in spans if "font_ai" in s.get("style", {})]
        print("total_spans=", len(spans))
        print("font_ai_tagged=", len(tagged))
        if tagged:
            st = tagged[0]["style"]
            print("sample_font_ai=", st.get("font_ai"))
            print("sample_score=", st.get("font_ai_score"))

asyncio.run(main())
PY

log "ALL DONE"
