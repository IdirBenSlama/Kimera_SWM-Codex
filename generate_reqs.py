"""
Scans the project to generate a comprehensive requirements.txt file.
"""
import os
import re
import pkgutil

# Get a list of standard library modules
STD_LIBS = {mod.name for mod in pkgutil.iter_modules()}
# Add some modules that are often part of Python but not always detected
STD_LIBS.update([
    'asyncio', 'collections', 'dataclasses', 'datetime', 'enum',
    'functools', 'importlib', 'json', 'logging', 'math', 'os',
    'pathlib', 're', 'sys', 'time', 'traceback', 'typing', 'unittest', 'uuid'
])

IMPORT_REGEX = re.compile(r"^\s*(?:from|import)\s+([a-zA-Z0-9_]+)")

def get_base_package(import_statement):
    """Extracts the base package from an import statement."""
    match = IMPORT_REGEX.match(import_statement)
    if match:
        return match.group(1)
    return None

def main():
    """Main function to generate requirements."""
    source_files = []
    with open("all_python_files.txt", "r", encoding="utf-8") as f:
        for line in f:
            path = line.strip()
            # Exclude virtual environment, archives, and tests from scan
            if ".venv" not in path and "archive" not in path and "tests" not in path:
                if os.path.isfile(path):
                    source_files.append(path)

    dependencies = set()
    for file_path in source_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    # Ignore comments
                    if line.strip().startswith("#"):
                        continue
                    
                    base_package = get_base_package(line)
                    if base_package:
                        # Exclude local project imports
                        if base_package not in ["backend", "scientific"] and base_package not in STD_LIBS:
                            dependencies.add(base_package)
        except Exception as e:
            print(f"Could not process {file_path}: {e}")

    # Add packages that might be missed by the simple regex
    # These were found in the error logs
    missed_packages = ["aiofiles", "yaml", "torch", "numpy", "pytest", "uvicorn", "pydantic", "sqlalchemy", "fastapi"]
    for pkg in missed_packages:
        dependencies.add(pkg)

    print("Found dependencies:", sorted(list(dependencies)))

    # Write to requirements.txt
    with open("requirements.txt", "w", encoding="utf-8") as f:
        for dep in sorted(list(dependencies)):
            f.write(f"{dep}\n")
    
    print("\nrequirements.txt has been generated.")

if __name__ == "__main__":
    main() 