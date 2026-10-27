"""The windowless starter that launch_training.ps1 registers as a task action."""
from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from pathlib import Path

import pytest

STARTER = Path(__file__).resolve().parents[1] / "scripts" / "run_without_window.py"

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="the starter is Windows-only")


def _exited_within(pid: int, seconds: float) -> bool:
    synchronize = 0x00100000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    handle = kernel32.OpenProcess(synchronize, False, pid)
    if not handle:
        return True
    try:
        wait_object_0 = 0
        return kernel32.WaitForSingleObject(ctypes.c_void_p(handle), int(seconds * 1000)) == wait_object_0
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


def test_starter_returns_the_program_exit_code(tmp_path):
    errors = tmp_path / "errors.txt"

    result = subprocess.run(
        [sys.executable, str(STARTER), str(errors), "cmd.exe", "/c", "exit 7"], timeout=60,
    )

    assert result.returncode == 7
    assert not errors.exists()


def test_starter_reports_a_program_it_cannot_start(tmp_path):
    errors = tmp_path / "errors.txt"

    result = subprocess.run(
        [sys.executable, str(STARTER), str(errors), str(tmp_path / "missing.exe")], timeout=60,
    )

    assert result.returncode == 1
    assert "FileNotFoundError" in errors.read_text(encoding="utf-8")


def test_ending_the_starter_ends_the_program(tmp_path):
    pid_file = tmp_path / "pid.txt"
    program = (
        "import os, sys, time; "
        "open(sys.argv[1], 'w').write(str(os.getpid())); "
        "time.sleep(120)"
    )
    starter = subprocess.Popen(
        [sys.executable, str(STARTER), str(tmp_path / "errors.txt"), sys.executable, "-c", program, str(pid_file)],
    )
    try:
        deadline = time.monotonic() + 60
        while not pid_file.exists() or not pid_file.read_text():
            assert time.monotonic() < deadline, "the program did not start"
            time.sleep(0.1)
        program_pid = int(pid_file.read_text())
        assert not _exited_within(program_pid, 0)

        starter.kill()
        starter.wait(timeout=30)

        assert _exited_within(program_pid, 10)
    finally:
        if starter.poll() is None:
            starter.kill()
