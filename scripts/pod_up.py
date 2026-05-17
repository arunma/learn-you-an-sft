"""Create a RunPod pod, provision the project, and copy data files.

Reads .env for:
  RUNPOD_API_KEY   (required)
  HF_TOKEN, HF_PUSH_REPO, ANTHROPIC_API_KEY  (forwarded to pod env)

Workflow:
  1. Create pod via RunPod API with sensible defaults (A6000, 48GB VRAM)
  2. Wait for RUNNING + SSH port assignment
  3. Wait for SSH service to accept connections
  4. Clone repo + uv sync on pod (unless --skip-setup)
  5. scp data/processed/{train,val,manifest} (unless --skip-data)
  6. Save pod_id to .runpod_pod_id for pod_down
  7. Print SSH + tunnel commands

Usage:
  uv add runpod                                       # one-time
  uv run python -m scripts.pod_up                     # defaults
  uv run python -m scripts.pod_up --gpu "NVIDIA RTX 4090"
  uv run python -m scripts.pod_up --skip-data
  uv run python -m scripts.pod_up --skip-setup
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

try:
    import runpod
except ImportError:
    raise SystemExit("runpod SDK not installed. Run: uv add runpod")


REPO_URL = "https://github.com/arunma/learn-you-an-sft.git"
REPO_DIR_ON_POD = "/workspace/learn-you-an-sft"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_GPU = "NVIDIA RTX A6000"
DEFAULT_DISK_GB = 50
POD_NAME = "learn-you-an-sft"
STATE_FILE = Path(".runpod_pod_id")

FORWARD_ENV_VARS = ("HF_TOKEN", "HF_PUSH_REPO", "ANTHROPIC_API_KEY")

LOCAL_DATA_FILES = (
    "data/processed/train.jsonl",
    "data/processed/val.jsonl",
    "data/processed/manifest.json",
)


def _wait_for_running(pod_id: str, timeout: int = 300) -> tuple[str | None, int | None]:
    """Block until pod is RUNNING with SSH port assigned. Returns (host, port) or (None, None)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = runpod.get_pod(pod_id)
        if status.get("desiredStatus") == "RUNNING" and status.get("runtime"):
            ports = status["runtime"].get("ports") or []
            for p in ports:
                if p.get("privatePort") == 22 and p.get("isIpPublic"):
                    return p["ip"], int(p["publicPort"])
        time.sleep(5)
    return None, None


def _wait_for_ssh(host: str, port: int, ssh_key: str, timeout: int = 180) -> bool:
    """Block until SSH actually accepts connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(
            [
                "ssh",
                "-i", ssh_key,
                "-p", str(port),
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ConnectTimeout=5",
                "-o", "BatchMode=yes",
                "-o", "LogLevel=ERROR",
                f"root@{host}",
                "echo ready",
            ],
            capture_output=True,
            timeout=15,
        )
        if result.returncode == 0:
            return True
        time.sleep(3)
    return False


def _build_setup_script() -> str:
    # NOTE: RunPod's PyTorch images are minimal — no uv, no tmux. Install both
    # before using them. The astral installer writes uv to ~/.local/bin and
    # updates ~/.bashrc so future logins pick it up automatically.
    return f"""set -e
# Install tmux (and curl, just in case) via apt
if ! command -v tmux >/dev/null 2>&1; then
    echo ">>> Installing tmux + curl via apt..."
    apt-get update -qq
    apt-get install -y -qq tmux curl
fi
# Install uv via the astral installer
if ! command -v uv >/dev/null 2>&1; then
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version
cd /workspace
if [ -d learn-you-an-sft ]; then
    cd learn-you-an-sft && git pull
else
    git clone {REPO_URL}
    cd learn-you-an-sft
fi
mkdir -p data/processed
uv sync
# PyPI's default torch wheels are compiled for cu126/cu128. The RunPod
# PyTorch 2.4 image ships CUDA 12.4 drivers. We pin torch to 2.5.1+cu124
# because:
#   - 2.5.1 is stable on cu124 (the RunPod image's driver)
#   - 2.6.0+cu124 needs NCCL 2.23+ symbols that aren't always available
# Install with full deps so nvidia-cudnn / nvidia-cuda-runtime versions
# match what this torch was compiled against.
echo ">>> Pinning torch 2.5.1+cu124 (matches RunPod image driver + NCCL)..."
# Use `uv pip` instead of `.venv/bin/pip`: uv-managed venvs don't ship pip
# by default. `uv pip install` is uv's pip-equivalent and works in any
# uv venv without bootstrapping. Note: this DOES NOT trigger uv sync —
# it's a deliberate install. To prevent `uv run` from later reverting
# this pin, use `uv run --no-sync` or `.venv/bin/python` directly.
uv pip install --force-reinstall \\
    "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu124
echo "SETUP DONE"
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--gpu", default=DEFAULT_GPU, help=f"GPU type id (default: {DEFAULT_GPU!r})")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Container image")
    parser.add_argument(
        "--ssh-key",
        default=os.path.expanduser("~/.ssh/id_ed25519_arunma"),
        help="SSH private key path",
    )
    parser.add_argument("--name", default=POD_NAME, help="Pod name")
    parser.add_argument("--disk-gb", type=int, default=DEFAULT_DISK_GB, help="Container disk GB")
    parser.add_argument("--skip-data", action="store_true", help="Skip scp of data/processed files")
    parser.add_argument("--skip-setup", action="store_true", help="Skip git clone + uv sync on pod")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY not set (add to .env or export)")
    runpod.api_key = api_key

    env_to_forward = {k: v for k in FORWARD_ENV_VARS if (v := os.environ.get(k))}
    if "HF_TOKEN" not in env_to_forward:
        print("WARN: HF_TOKEN not set — pulling private adapters from HF will fail.")
    if "ANTHROPIC_API_KEY" not in env_to_forward:
        print("WARN: ANTHROPIC_API_KEY not set — eval.run_eval Haiku calls will fail.")

    ssh_key_path = Path(args.ssh_key)
    if not ssh_key_path.exists():
        raise SystemExit(f"SSH key not found: {ssh_key_path}")

    print(f"Creating pod: {args.name!r} ({args.gpu})")
    print(f"  Image: {args.image}")
    print(f"  Disk:  {args.disk_gb} GB")
    print(f"  Forwarding env: {sorted(env_to_forward.keys()) or '(none)'}")

    pod = runpod.create_pod(
        name=args.name,
        image_name=args.image,
        gpu_type_id=args.gpu,
        cloud_type="ALL",
        container_disk_in_gb=args.disk_gb,
        volume_in_gb=0,
        ports="22/tcp",
        env=env_to_forward,
    )
    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")
    STATE_FILE.write_text(pod_id)
    print(f"  pod_id saved to {STATE_FILE}")

    print("Waiting for pod to reach RUNNING + expose SSH port (up to 5 min)...")
    host, port = _wait_for_running(pod_id)
    if not host or not port:
        raise SystemExit(
            f"Pod {pod_id} didn't reach RUNNING in 5 min. Check the RunPod dashboard."
        )
    print(f"  SSH endpoint: {host}:{port}")

    print("Waiting for SSH service to accept connections (up to 3 min)...")
    if not _wait_for_ssh(host, port, args.ssh_key):
        print("SSH didn't come up in time. Pod is alive — try connecting manually:")
        print(f"  ssh -i {args.ssh_key} -p {port} root@{host}")
        return 1
    print("  SSH ready.")

    ssh_target = f"root@{host}"
    ssh_cmd = [
        "ssh",
        "-i", args.ssh_key,
        "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        ssh_target,
    ]
    scp_cmd_base = [
        "scp",
        "-i", args.ssh_key,
        "-P", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
    ]

    setup_ok = True
    if not args.skip_setup:
        print("Cloning repo + uv sync on pod...")
        result = subprocess.run(ssh_cmd + [_build_setup_script()], check=False)
        if result.returncode != 0:
            print(
                "WARN: setup script failed (exit "
                f"{result.returncode}). Continuing to data copy. "
                "Pod stays alive — fix via SSH."
            )
            setup_ok = False

    if not args.skip_data:
        existing = [f for f in LOCAL_DATA_FILES if Path(f).exists()]
        missing = [f for f in LOCAL_DATA_FILES if not Path(f).exists()]
        if missing:
            print(f"WARN: missing locally (skipped): {missing}")
        if existing:
            print(f"Copying data files: {existing}")
            subprocess.run(
                scp_cmd_base + existing + [f"{ssh_target}:{REPO_DIR_ON_POD}/data/processed/"],
                check=True,
            )

    print()
    print("=" * 66)
    print(f"POD READY: {pod_id}")
    if not setup_ok:
        print("WARN: setup step failed earlier — SSH in and finish manually.")
    print()
    print(f"SSH:    ssh -i {args.ssh_key} -p {port} {ssh_target}")
    print(f"TUNNEL: ssh -i {args.ssh_key} -p {port} -L 6006:localhost:6006 {ssh_target} -N")
    print()
    print("ON THE POD — use --no-sync so uv doesn't revert the pinned torch:")
    print("  uv run --no-sync python -m runs.sft_v1_trl.train")
    print("  uv run --no-sync python -m eval.run_eval --concurrency 5")
    print()
    print("DESTROY: uv run python -m scripts.pod_down")
    print("=" * 66)
    return 0 if setup_ok else 1


if __name__ == "__main__":
    sys.exit(main())
