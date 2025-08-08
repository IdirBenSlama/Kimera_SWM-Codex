import argparse
import ast
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple
import tokenize

# Ensure UTF-8 output to avoid Windows console encoding issues
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def discover_python_files(root_dir: str) -> List[str]:
    python_files: List[str] = []
    for current_root, dirnames, filenames in os.walk(root_dir):
        # Skip common noisy or irrelevant directories
        dirnames[:] = [d for d in dirnames if d not in {".venv", "venv", "__pycache__", ".git"}]
        for filename in filenames:
            if filename.endswith(".py"):
                python_files.append(os.path.join(current_root, filename))
    return python_files


def check_file_syntax(file_path: str) -> Optional[Tuple[str, int, int, str]]:
    try:
        with tokenize.open(file_path) as f:  # respects PEP 263 encoding declarations
            source = f.read()
        ast.parse(source, filename=file_path, mode="exec")
        return None
    except SyntaxError as exc:  # Return concise, actionable info
        line = exc.lineno or 0
        col = exc.offset or 0
        msg = exc.msg or "invalid syntax"
        return (file_path, line, col, msg)
    except Exception as exc:
        # Treat unexpected parsing errors as actionable diagnostics as well
        return (file_path, 0, 0, f"non-syntax error during parse: {type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast parallel AST-based syntax scan (no pyc writes)")
    parser.add_argument("--root", default="src", help="Root directory to scan (default: src)")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2), help="Parallel workers")
    args = parser.parse_args()

    files = discover_python_files(args.root)
    if not files:
        print(f"No Python files found under {args.root}")
        return 0

    print(f"Scanning {len(files)} files under {args.root} with {args.jobs} workers...", flush=True)

    errors: List[Tuple[str, int, int, str]] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        future_to_file = {executor.submit(check_file_syntax, f): f for f in files}
        for future in as_completed(future_to_file):
            result = future.result()
            if result is not None:
                errors.append(result)

    if errors:
        # Stable sort by file, then line
        errors.sort(key=lambda e: (e[0], e[1], e[2]))
        print("\nSyntax errors detected:", flush=True)
        for file_path, line, col, msg in errors:
            location = f"{file_path}:{line}:{col}" if line or col else file_path
            try:
                print(f"{location}: {msg}")
            except UnicodeEncodeError:
                safe = f"{location}: {msg}".encode("utf-8", errors="replace").decode("utf-8", errors="replace")
                print(safe)
        print(f"\nTotal files with errors: {len({e[0] for e in errors})}")
        print(f"Total syntax issues: {len(errors)}")
        return 1
    else:
        print("No syntax errors detected.")
        return 0


if __name__ == "__main__":
    sys.exit(main())


