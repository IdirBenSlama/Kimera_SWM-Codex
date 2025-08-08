#!/usr/bin/env python3
import ast
import os
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Set, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
REQ_DIR = PROJECT_ROOT / "requirements_consolidated"

# Map submodules to top-level package names
MODULE_ALIASES = {
    # Common mappings
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "Crypto": "pycryptodome",
}

# Ignore stdlib (rough list augmentation can be added if needed)
STDLIB_HINT = set([
    "os","sys","re","json","logging","typing","asyncio","pathlib","datetime","time","math","functools","itertools","collections",
    "subprocess","argparse","dataclasses","enum","queue","hashlib","base64","random","statistics","inspect","threading","contextlib",
])


def parse_imports(py_file: Path) -> Tuple[Set[str], List[str]]:
    pkgs: Set[str] = set()
    syntax_errors: List[str] = []
    try:
        content = py_file.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(content, filename=str(py_file))
    except SyntaxError as e:
        syntax_errors.append(f"{py_file}: {e.msg} at line {e.lineno}")
        return pkgs, syntax_errors
    except Exception as e:
        syntax_errors.append(f"{py_file}: {e}")
        return pkgs, syntax_errors

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                top = n.name.split(".")[0]
                if top not in STDLIB_HINT:
                    pkgs.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top not in STDLIB_HINT:
                    pkgs.add(top)
    return pkgs, syntax_errors


def scan_src() -> Tuple[Set[str], List[str]]:
    all_pkgs: Set[str] = set()
    all_errors: List[str] = []
    for py_file in SRC_DIR.rglob("*.py"):
        # Skip backups
        if py_file.suffix.endswith(".backup_cleanup_fix"):
            continue
        pkgs, errs = parse_imports(py_file)
        all_pkgs |= pkgs
        all_errors.extend(errs)
    return all_pkgs, all_errors


def map_alias(pkg: str) -> str:
    return MODULE_ALIASES.get(pkg, pkg)


def load_requirements() -> Set[str]:
    reqs: Set[str] = set()
    if REQ_DIR.exists():
        for req_file in [REQ_DIR / "base.txt", REQ_DIR / "api.txt"]:
            if req_file.exists():
                for line in req_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    name = line.split("==")[0].strip()
                    reqs.add(name.lower())
    return reqs


def check_installed(top_packages: Set[str]) -> Dict[str, str]:
    status: Dict[str, str] = {}
    for pkg in sorted(top_packages):
        top = map_alias(pkg)
        try:
            spec = importlib.util.find_spec(pkg)
            if spec is not None:
                status[top] = "installed"
            else:
                status[top] = "missing"
        except Exception:
            status[top] = "missing"
    return status


def main() -> None:
    print("=== Static Dependency & Syntax Scan ===")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Source:  {SRC_DIR}")

    pkgs, errors = scan_src()
    reqs = load_requirements()
    installed = check_installed(pkgs)

    missing = [p for p, st in installed.items() if st == "missing"]
    not_in_requirements = [p for p in pkgs if map_alias(p).lower() not in reqs]

    print("\n-- Syntax Errors --")
    if errors:
        for e in errors[:50]:
            print("  ", e)
        if len(errors) > 50:
            print(f"  ... and {len(errors)-50} more")
    else:
        print("  none")

    print("\n-- Top-level Imports Detected --")
    print("  ", ", ".join(sorted(map(map_alias, pkgs))) or "none")

    print("\n-- Missing (not installed in current interpreter) --")
    print("  ", ", ".join(sorted(missing)) or "none")

    print("\n-- Not Listed in requirements_consolidated --")
    print("  ", ", ".join(sorted(map(str.lower, not_in_requirements))) or "none")

    if missing:
        print("\nSuggested install:")
        print("  .\\.venv\\Scripts\\python.exe -m pip install " + " ".join(sorted(missing)))

if __name__ == "__main__":
    sys.exit(main())
