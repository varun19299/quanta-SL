"""
Delete all */__pycache__/* files

Useful for forcing numba to recompile JIT functions
"""
from pathlib import Path
import logging

def kill_numba_cache(root_folder: Path = Path(".")):
    for path in root_folder.iterdir():
        if path.is_dir() and path.name == "__pycache__":
            print(path)
            logging.debug(f"Empyting __pycache__ files from {path}")
            for file in path.iterdir():
                file.unlink()
        elif path.is_dir():
            kill_numba_cache(path)


if __name__ == "__main__":
    kill_numba_cache()
