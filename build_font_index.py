import argparse

from font_ai_matcher import FontAIMatcher


def main():
    ap = argparse.ArgumentParser(description="Validate Storia matcher and refresh local font catalog.")
    ap.add_argument("--model", default="./ai_models/fonts/teacher/model.onnx", help="Storia ONNX model path")
    ap.add_argument("--index", default="", help="Deprecated (kept for backward compatibility)")
    ap.add_argument("--max-fonts", type=int, default=0, help="Deprecated (kept for backward compatibility)")
    args = ap.parse_args()

    matcher = FontAIMatcher(model_path=args.model, index_path=args.index)
    if not matcher.has_model():
        raise SystemExit("Model not loaded. Provide a valid --model ONNX path.")

    matcher.rebuild_index(max_fonts=args.max_fonts)
    if not matcher.is_ready():
        raise SystemExit("Storia matcher not ready (check model/config/mapping files).")
    print(f"[OK] Storia matcher ready with {len(matcher._index_names)} classes.")


if __name__ == "__main__":
    main()
