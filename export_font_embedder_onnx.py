#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch

from train_font_embedder import SmallFontEmbedder, export_onnx


def main():
    ap = argparse.ArgumentParser(description="Export trained font embedder checkpoint to ONNX.")
    ap.add_argument("--checkpoint", default="./ai_models/fonts/font_embedder.pt", help="Checkpoint .pt path")
    ap.add_argument("--output", default="./ai_models/fonts/font_embedder.onnx", help="ONNX output path")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = SmallFontEmbedder(
        embedding_dim=int(ckpt["embedding_dim"]),
        variant=str(ckpt.get("model_variant", "small")),
    )
    model.load_state_dict(ckpt["model_state"])
    export_onnx(model, Path(args.output), image_size=int(ckpt["image_size"]))
    print(f"[OK] ONNX exported: {args.output}")


if __name__ == "__main__":
    main()
