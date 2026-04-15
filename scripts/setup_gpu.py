"""GPU setup script for installing the correct PyTorch build.

Usage:
    python scripts/setup_gpu.py rocm     # AMD GPUs (ROCm via TheRock)
    python scripts/setup_gpu.py cuda     # NVIDIA GPUs (CUDA 12.8)
    python scripts/setup_gpu.py cpu      # CPU-only (PyPI default)
    python scripts/setup_gpu.py detect   # Auto-detect and install

uv's lockfile resolver cannot cross-resolve TheRock/CUDA nightly indexes
with PyPI, so GPU torch must be installed separately via uv pip install.
This script wraps that into a single command.

NOTE: Running 'uv sync' will reinstall CPU torch from the lockfile,
overwriting the GPU build. Rerun this script after any 'uv sync'.
"""

import argparse
import subprocess
import sys
import shutil

GPU_CONFIGS = {
    "rocm": {
        "label": "AMD ROCm (TheRock)",
        "index_url": "https://rocm.nightlies.amd.com/v2/gfx120X-all/",
        "pre": True,
    },
    "cuda": {
        "label": "NVIDIA CUDA 12.8",
        "index_url": "https://download.pytorch.org/whl/nightly/cu128",
        "pre": True,
    },
    "cpu": {
        "label": "CPU-only (PyPI)",
        "index_url": None,
        "pre": False,
    },
}


def detect_gpu() -> str:
    """Detect GPU vendor using system tools. Conservative: falls back to cpu if uncertain."""
    # Check for NVIDIA via nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("Detected NVIDIA GPU via nvidia-smi")
                return "cuda"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # Check for AMD ROCm tools
    if shutil.which("rocm-smi") or shutil.which("rocminfo"):
        print("Detected AMD GPU via ROCm tools")
        return "rocm"

    # Check Windows for GPU names via WMI
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
                    print("WARNING: ROCm/TheRock support is limited to specific AMD GPUs (RDNA 3/4).")
                    print("         If installation fails, use 'cpu' instead.")
                    return "rocm"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    print("No supported GPU detected, defaulting to CPU")
    return "cpu"


def install_torch(gpu_type: str) -> int:
    """Install the appropriate torch build into the project venv."""
    config = GPU_CONFIGS[gpu_type]
    print(f"\n{'=' * 50}")
    print(f"Installing PyTorch for: {config['label']}")
    print(f"{'=' * 50}\n")

    cmd = ["uv", "pip", "install", "--reinstall", "--python", sys.executable, "torch"]
    if config["index_url"]:
        cmd.extend(["--index-url", config["index_url"]])
    if config["pre"]:
        cmd.append("--pre")

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode


def verify_install():
    """Verify the torch installation and report GPU status."""
    print(f"\n{'=' * 50}")
    print("Verifying installation...")
    print(f"{'=' * 50}\n")

    result = subprocess.run(
        [sys.executable, "-c", """
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
        epilog="Run 'uv sync' first to install all other dependencies.",
    )
    parser.add_argument(
        "gpu",
        choices=["rocm", "cuda", "cpu", "detect"],
        help="GPU type: rocm (AMD), cuda (NVIDIA), cpu (no GPU), detect (auto-detect)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the install command without running it",
    )
    args = parser.parse_args()

    gpu_type = args.gpu
    if gpu_type == "detect":
        gpu_type = detect_gpu()
        print(f"Detected GPU type: {gpu_type}")

    if args.dry_run:
        config = GPU_CONFIGS[gpu_type]
        cmd = ["uv", "pip", "install", "--reinstall", "--python", sys.executable, "torch"]
        if config["index_url"]:
            cmd.extend(["--index-url", config["index_url"]])
        if config["pre"]:
            cmd.append("--pre")
        print(f"Would run: {' '.join(cmd)}")
        return

    rc = install_torch(gpu_type)
    if rc != 0:
        print(f"\nInstallation failed with exit code {rc}", file=sys.stderr)
        sys.exit(rc)

    verify_install()


if __name__ == "__main__":
    main()
