from __future__ import annotations

import os
import site
from pathlib import Path


def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    for base in list(site.getsitepackages()) + [site.getusersitepackages()]:
        torch_lib = Path(base) / "torch" / "lib"
        if torch_lib.exists():
            dirs.append(torch_lib)
    return dirs


for dll_dir in _candidate_dirs():
    os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(dll_dir))
        except OSError:
            pass