"""Source revision recorded by journaled measurements."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from src.utils.source_revision import source_revision

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_dirty_trees_compare_equal_only_with_identical_edits(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "test")
    tracked = tmp_path / "module.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "module.py")
    _git(tmp_path, "commit", "-q", "-m", "initial")

    clean = source_revision(str(tmp_path))
    assert len(clean["commit"]) == 40 and clean["dirty"] is False

    (tmp_path / "untracked.txt").write_text("ignored\n", encoding="utf-8")
    assert source_revision(str(tmp_path)) == clean

    tracked.write_text("VALUE = 2\n", encoding="utf-8")
    first = source_revision(str(tmp_path))
    tracked.write_text("VALUE = 3\n", encoding="utf-8")
    second = source_revision(str(tmp_path))
    assert first["commit"] == second["commit"] == clean["commit"]
    assert isinstance(first["dirty"], str) and isinstance(second["dirty"], str)
    assert first["dirty"] != second["dirty"]

    tracked.write_text("VALUE = 2\n", encoding="utf-8")
    assert source_revision(str(tmp_path)) == first


def test_missing_repository_records_nothing(tmp_path):
    assert source_revision(str(tmp_path / "absent")) == {"commit": None, "dirty": None}
