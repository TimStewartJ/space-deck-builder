"""GPU setup script for installing the correct PyTorch build.

Usage:
    python scripts/setup_gpu.py rocm     # AMD GPUs (ROCm via TheRock)
    python scripts/setup_gpu.py cuda     # NVIDIA GPUs (CUDA 12.8)
    python scripts/setup_gpu.py cpu      # CPU-only (PyPI default)
    python scripts/setup_gpu.py detect   # Auto-detect and install

Wraps 'uv sync --extra <backend>' to install the correct PyTorch variant.
GPU backends are defined as optional-dependency extras in pyproject.toml,
with [tool.uv.sources] routing each to the appropriate package index.
"""

import argparse
import subprocess
import sys
import shutil

GPU_LABELS = {
    "rocm": "AMD ROCm (TheRock)",
    "cuda": "NVIDIA CUDA 12.8",
    "cpu":  "CPU-only (PyPI)",
}


def detect_gpu() -> str:
    """Detect GPU vendor using system tools. Conservative: falls back to cpu if uncertain."""
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("Detected NVIDIA GPU via nvidia-smi")
                return "cuda"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    if shutil.which("rocm-smi") or shutil.which("rocminfo"):
        print("Detected AMD GPU via ROCm tools")
        return "rocm"

    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["powershell", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                output = result.stdout.lower()
                if "nvidia" in output or "geforce" in output or "rtx" in output or "quadro" in output:
                    print(f"Detected NVIDIA GPU via WMI: {result.stdout.strip()}")
                    return "cuda"
                if "radeon" in output:
                    print(f"Detected AMD GPU via WMI: {result.stdout.strip()}")
                    return "rocm"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    print("No supported GPU detected, defaulting to CPU")
    return "cpu"


def sync_backend(gpu_type: str) -> int:
    """Run 'uv sync --extra <backend>' to install the correct PyTorch."""
    label = GPU_LABELS[gpu_type]
    print(f"\n{'=' * 50}")
    print(f"Installing PyTorch for: {label}")
    print(f"{'=' * 50}\n")

    cmd = ["uv", "sync", "--extra", gpu_type]
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode


def verify_install(gpu_type: str):
    """Verify the torch installation in the project venv and report GPU status."""
    print(f"\n{'=' * 50}")
    print("Verifying installation...")
    print(f"{'=' * 50}\n")

    result = subprocess.run(
        ["uv", "run", "--extra", gpu_type, "python", "-c", """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device:      {torch.cuda.get_device_name(0)}")
    print(f"GPU arch:        {torch.cuda.get_device_capability(0)}")
else:
    print("GPU device:      None (CPU-only)")
"""],
        capture_output=False,
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Install the correct PyTorch build for your GPU.",
    )
    parser.add_argument(
        "gpu",
        choices=["rocm", "cuda", "cpu", "detect"],
        help="GPU type: rocm (AMD), cuda (NVIDIA), cpu (no GPU), detect (auto-detect)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sync command without running it",
    )
    args = parser.parse_args()

    gpu_type = args.gpu
    if gpu_type == "detect":
        gpu_type = detect_gpu()
        print(f"Detected GPU type: {gpu_type}")

    if args.dry_run:
        print(f"Would run: uv sync --extra {gpu_type}")
        return

    rc = sync_backend(gpu_type)
    if rc != 0:
        print(f"\nInstallation failed with exit code {rc}", file=sys.stderr)
        sys.exit(rc)

    rc = verify_install(gpu_type)
    if rc != 0:
        print(f"\nVerification failed with exit code {rc}", file=sys.stderr)
        sys.exit(rc)


if __name__ == "__main__":
    main()
