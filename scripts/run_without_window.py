"""Run a console program without a window, tied to this process's lifetime.

launch_training.ps1 registers this file as a scheduled task's action, run by
the GUI-subsystem pythonw.exe, which has no console. It first joins a Job
Object that terminates every process in it when its last handle closes, then
starts its arguments with CREATE_NO_WINDOW, so the program gets a console
that has no window, and exits with the program's exit code. Ending this
process, as Task Scheduler does when the task is stopped, ends the whole
process tree.

Usage: pythonw.exe run_without_window.py <error file> <program> [arguments...]

pythonw has no console, so a failure to start is written to <error file>.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
import traceback
from ctypes import wintypes

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def join_kill_on_close_job() -> int:
    """Put this process in a new kill-on-close job and return its handle.

    The handle stays open until this process ends; closing it then
    terminates every process still in the job, nested jobs included.
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = _ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())
    return job


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        return 2
    error_file, command = argv[1], argv[2:]
    try:
        if not command:
            raise ValueError("no program to run")
        job = join_kill_on_close_job()  # noqa: F841 - held open until exit
        process = subprocess.Popen(command, creationflags=subprocess.CREATE_NO_WINDOW)
    except BaseException:
        with open(error_file, "w", encoding="utf-8") as fh:
            traceback.print_exc(file=fh)
        return 1
    return process.wait()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
