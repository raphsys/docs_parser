import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translator import DocumentTranslator


def main():
    root = Path(__file__).resolve().parent
    cases_path = root / "style_tone_regression_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    tr = DocumentTranslator()
    for case in cases:
        text = case["text"]
        print("SRC:", text)
        for profile in case.get("profiles", []):
            out = tr.translate_text(
                text,
                target_lang=case.get("target_lang", "fr"),
                block_role=case.get("block_role", "body"),
                strategy=case.get("strategy", "semantic_reflow"),
                style=profile.get("style"),
                tone=profile.get("tone"),
            )
            print(profile.get("style"), profile.get("tone"), "=>", out)
        print("---")


if __name__ == "__main__":
    main()
