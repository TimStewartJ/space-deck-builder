"""The git revision of the source tree a measurement runs from.

Journaled measurements (the held-out league and the purchase audit) record it
so that resuming never mixes results produced by different code.
"""
from __future__ import annotations

import os
import subprocess

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def source_revision(root: str = REPOSITORY_ROOT) -> dict:
    """The checked-out commit and whether tracked files differ from it.

    Both values are ``None`` when git or the repository is unavailable.
    """
    try:
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True, timeout=120,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}
