#!/usr/bin/env python3
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
old = s

needle = "from source_ownership import audit_source_ownership, build_source_ownership"
bootstrap = '# Allow direct execution from tools/:\n#   python tools/audit_source_ownership.py results/<run>\n_PROJECT_ROOT = Path(__file__).resolve().parents[1]\nif str(_PROJECT_ROOT) not in sys.path:\n    sys.path.insert(0, str(_PROJECT_ROOT))\n\nfrom source_ownership import audit_source_ownership, build_source_ownership'

if needle in s and "_PROJECT_ROOT = Path(__file__).resolve().parents[1]" not in s:
    s = s.replace(needle, bootstrap, 1)

if s != old:
    p.write_text(s, encoding="utf-8")
    print(f"corrigé: {p}")
else:
    print(f"aucune modification nécessaire: {p}")
