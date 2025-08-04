import os
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    filename='find_largest_files.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def find_largest_files(directory: str, top_n: int = 20) -> List[Tuple[str, int]]:
    """
    Recursively find the top N largest files in the given directory.

    :param directory: The root directory to scan.
    :param top_n: Number of largest files to return.
    :return: List of tuples (file_path, file_size_bytes) sorted by size descending.
    :raises Exception: If directory cannot be accessed.
    """
    file_sizes = []
    for root, _, files in os.walk(directory):
        for file in files:
            try:
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                file_sizes.append((file_path, size))
            except (OSError, PermissionError) as e:
                logging.warning(f"Could not access {file_path}: {e}")
    file_sizes.sort(key=lambda x: x[1], reverse=True)
    return file_sizes[:top_n]

def write_report(largest_files: List[Tuple[str, int]], report_path: str) -> None:
    """
    Write the largest files report to a text file.

    :param largest_files: List of (file_path, file_size_bytes) tuples.
    :param report_path: Path to the output report file.
    """
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Top Largest Files in Project:\n\n")
            for path, size in largest_files:
                f.write(f"{size/1024/1024:.2f} MB\t{path}\n")
        logging.info(f"Report written to {report_path}")
    except Exception as e:
        logging.error(f"Failed to write report: {e}")

def main() -> None:
    """
    Main function to find and report the largest files in the project.
    """
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        largest_files = find_largest_files(project_root, top_n=20)
        write_report(largest_files, os.path.join(project_root, 'largest_files_report.txt'))
        logger.info("Scan complete. See 'largest_files_report.txt' and 'find_largest_files.log' for details.")
    except Exception as e:
        logging.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main() 