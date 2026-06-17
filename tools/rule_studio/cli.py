"""CLI Rule Studio : scan / export / audit sans interface graphique.

Utile pour CI ou usage rapide :
    python -m tools.rule_studio.cli scan
    python -m tools.rule_studio.cli audit --out RULE_AUDIT_REPORT.md
    python -m tools.rule_studio.cli export --format csv --out rules.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.rule_studio.core.studio import Studio  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rule_studio")
    parser.add_argument("--root", default=str(_REPO_ROOT), help="Racine du projet")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("scan", help="Scanner le projet et peupler le store")

    p_audit = sub.add_parser("audit", help="Générer le rapport d'audit")
    p_audit.add_argument("--out", default="RULE_AUDIT_REPORT.md")
    p_audit.add_argument("--scan", action="store_true", help="Scanner avant")

    p_exp = sub.add_parser("export", help="Exporter l'inventaire")
    p_exp.add_argument("--format", choices=["csv", "json", "markdown"], default="csv")
    p_exp.add_argument("--out", required=True)
    p_exp.add_argument("--scan", action="store_true")

    args = parser.parse_args(argv)
    studio = Studio(args.root)

    if args.cmd == "scan":
        stats = studio.scan()
        print(f"Scan terminé : {stats}")
        return 0

    if getattr(args, "scan", False):
        studio.scan()

    if args.cmd == "audit":
        Path(args.out).write_text(studio.export_audit(), encoding="utf-8")
        print(f"Rapport écrit : {args.out}")
        return 0

    if args.cmd == "export":
        data = {
            "csv": studio.export_csv,
            "json": studio.export_json,
            "markdown": studio.export_markdown,
        }[args.format]()
        Path(args.out).write_text(data, encoding="utf-8")
        print(f"Export {args.format} écrit : {args.out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
