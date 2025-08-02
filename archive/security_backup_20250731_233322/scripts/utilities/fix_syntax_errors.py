import argparse
import os
import re

# This regex is designed to find logger calls that are likely to be broken.
# It looks for lines starting with 'logger.' and ending with something that isn't a closing parenthesis,
# potentially followed by whitespace.
# It's a heuristic and might not catch all cases, but it's a good starting point.
BROKEN_LOGGER_CALL_RE = re.compile(r"^\s*logger\.(info|warning|error|debug|critical)\((f?['\"].*)", re.DOTALL)

def fix_file(file_path):
    """
    Attempts to fix common syntax errors in a given Python file.
    Specifically targets unterminated f-strings in logger calls.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    original_lines = list(lines)
    lines_to_write = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Heuristic 1: Check for logger calls that don't end with ')' or '")'
        match = BROKEN_LOGGER_CALL_RE.match(line)
        if match and not line.strip().endswith(')'):
            
            # Simple case: the line is just missing a closing parenthesis
            if line.count('(') == line.count(')') + 1:
                 lines_to_write.append(line.rstrip() + ')\n')
                 i += 1
                 continue

            # Multi-line case
            temp_line = line.rstrip()
            paren_balance = temp_line.count('(') - temp_line.count(')')
            j = i + 1
            while j < len(lines) and paren_balance > 0:
                next_line = lines[j].rstrip()
                temp_line += " " + next_line.lstrip()
                paren_balance = temp_line.count('(') - temp_line.count(')')
                j += 1
            
            # After combining, if it's still unbalanced, add a closing parenthesis
            if temp_line.count('(') == temp_line.count(')') + 1:
                lines_to_write.append(temp_line + ')\n')
            else:
                 # If logic fails, just append original lines
                 lines_to_write.append(line)

            i = j

        else:
            lines_to_write.append(line)
            i += 1

    if lines_to_write != original_lines:
        print(f"Fixing {file_path}...")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines_to_write)
        except Exception as e:
            print(f"Error writing to {file_path}: {e}")


def main():
    """Main function to parse arguments and fix files."""
    parser = argparse.ArgumentParser(description="Fix common syntax errors in Python files.")
    parser.add_argument("file", help="The path to the Python file to fix.")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found at {args.file}")
        return

    fix_file(args.file)

if __name__ == "__main__":
    main() 