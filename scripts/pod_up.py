"""Create a RunPod pod, provision the project, scp data files.

Reads .env for:
  RUNPOD_API_KEY                                      (required, local-only)
  HF_TOKEN, HF_PUSH_REPO, ANTHROPIC_API_KEY           (forwarded to pod env)

Workflow:
  1. Create pod via RunPod API (defaults: RTX 4090, 24 GB VRAM)
  2. Wait for RUNNING + SSH port assignment
  3. Wait for SSH to accept connections
  4. Clone repo + uv sync on pod (unless --skip-setup)
  5. scp data/processed/val.jsonl to the pod (unless --skip-data)
  6. Save pod_id to .runpod_pod_id for pod_down
  7. Print SSH + tunnel commands

Defaults target the eval workflow. For training, override --gpu to an
A6000 / RTX 6000 Ada / L40S / H100 and pass --upload to add train.jsonl.

Usage:
  uv run python -m scripts.pod_up                       # defaults: RTX 4090, val.jsonl
  uv run python -m scripts.pod_up --gpu "NVIDIA RTX A6000"
  uv run python -m scripts.pod_up --upload data/processed/train.jsonl,data/processed/val.jsonl
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
    raise SystemExit("runpod SDK not installed. Run: uv sync")


REPO_URL = "https://github.com/arunma/learn-you-an-sft.git"
REPO_DIR_ON_POD = "/workspace/learn-you-an-sft"
DEFAULT_IMAGE = "pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel"
DEFAULT_GPU = "NVIDIA GeForce RTX 4090"
DEFAULT_DISK_GB = 50
POD_NAME = "learn-you-an-sft"
STATE_FILE = Path(".runpod_pod_id")

FORWARD_ENV_VARS = ("HF_TOKEN", "HF_PUSH_REPO", "ANTHROPIC_API_KEY")

DEFAULT_UPLOAD = ("data/processed/val.jsonl",)


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
    # RunPod's PyTorch images don't ship uv or tmux. Install both. The astral
    # installer writes uv to ~/.local/bin and patches ~/.bashrc.
    return f"""set -e
if ! command -v tmux >/dev/null 2>&1; then
    echo ">>> Installing tmux + curl via apt..."
    apt-get update -qq
    apt-get install -y -qq tmux curl
fi
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
mkdir -p data/processed runs/eval_reports
uv sync
echo ">>> Installing vLLM 0.7+ (supports Qwen3, brings its own torch deps)..."
uv pip install "vllm>=0.7.0"
echo "SETUP DONE"
"""


def provision_pod(
    gpu: str,
    image: str,
    disk_gb: int,
    name: str,
    ssh_key: str,
    upload: tuple[str, ...],
    skip_setup: bool,
    skip_data: bool,
) -> tuple[str, str, int]:
    """End-to-end provisioning. Returns (pod_id, host, port) on success.

    Raises SystemExit on any unrecoverable error. Caller is responsible for
    terminating the pod on failure (the orchestrator does this in a finally
    block; CLI users get the pod_id printed and call `pod_down` manually).
    """
    env_to_forward = {k: v for k in FORWARD_ENV_VARS if (v := os.environ.get(k))}
    if "HF_TOKEN" not in env_to_forward:
        print("WARN: HF_TOKEN not set — pulling private adapters from HF will fail.")
    if "ANTHROPIC_API_KEY" not in env_to_forward:
        print("WARN: ANTHROPIC_API_KEY not set — eval's Haiku calls will fail.")

    if not Path(ssh_key).exists():
        raise SystemExit(f"SSH key not found: {ssh_key}")

    print(f"Creating pod: {name!r} ({gpu})")
    print(f"  Image: {image}")
    print(f"  Disk:  {disk_gb} GB")
    print(f"  Forwarding env: {sorted(env_to_forward.keys()) or '(none)'}")

    pod = runpod.create_pod(
        name=name,
        image_name=image,
        gpu_type_id=gpu,
        cloud_type="ALL",
        container_disk_in_gb=disk_gb,
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
    if not _wait_for_ssh(host, port, ssh_key):
        raise SystemExit(
            f"SSH didn't come up in 3 min. Pod is alive — connect manually:\n"
            f"  ssh -i {ssh_key} -p {port} root@{host}"
        )
    print("  SSH ready.")

    ssh_target = f"root@{host}"
    ssh_cmd = [
        "ssh", "-i", ssh_key, "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        ssh_target,
    ]
    scp_cmd_base = [
        "scp", "-i", ssh_key, "-P", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
    ]

    if not skip_setup:
        print("Cloning repo + uv sync on pod...")
        result = subprocess.run(ssh_cmd + [_build_setup_script()], check=False)
        if result.returncode != 0:
            raise SystemExit(
                f"Setup script failed (exit {result.returncode}). "
                f"Pod {pod_id} is alive — investigate via SSH."
            )

    if not skip_data:
        existing = [f for f in upload if Path(f).exists()]
        missing = [f for f in upload if not Path(f).exists()]
        if missing:
            print(f"WARN: missing locally (skipped): {missing}")
        if existing:
            print(f"Copying data files: {existing}")
            subprocess.run(
                scp_cmd_base + existing + [f"{ssh_target}:{REPO_DIR_ON_POD}/data/processed/"],
                check=True,
            )

    return pod_id, host, port


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
    parser.add_argument(
        "--upload",
        default=",".join(DEFAULT_UPLOAD),
        help="Comma-separated list of files to scp into data/processed/ on the pod",
    )
    parser.add_argument("--skip-data", action="store_true", help="Skip data upload")
    parser.add_argument("--skip-setup", action="store_true", help="Skip git clone + uv sync")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY not set (add to .env)")
    runpod.api_key = api_key

    upload = tuple(p.strip() for p in args.upload.split(",") if p.strip())

    pod_id, host, port = provision_pod(
        gpu=args.gpu,
        image=args.image,
        disk_gb=args.disk_gb,
        name=args.name,
        ssh_key=args.ssh_key,
        upload=upload,
        skip_setup=args.skip_setup,
        skip_data=args.skip_data,
    )

    print()
    print("=" * 66)
    print(f"POD READY: {pod_id}")
    print()
    print(f"SSH:    ssh -i {args.ssh_key} -p {port} root@{host}")
    print(f"TUNNEL: ssh -i {args.ssh_key} -p {port} -L 6006:localhost:6006 root@{host} -N")
    print()
    print("ON THE POD — use --no-sync so uv doesn't revert the pinned torch:")
    print(f"  cd {REPO_DIR_ON_POD}")
    print("  uv run --no-sync python -m run eval")
    print("  uv run --no-sync python -m run train")
    print()
    print("DESTROY: uv run python -m scripts.pod_down")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
