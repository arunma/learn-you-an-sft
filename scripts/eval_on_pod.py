"""End-to-end: spin up a RunPod GPU, run `run eval`, scp reports back, terminate.

  uv run python -m scripts.eval_on_pod                  # walk a GPU candidate list
  uv run python -m scripts.eval_on_pod --gpu "NVIDIA RTX A6000"  # force single GPU
  uv run python -m scripts.eval_on_pod --keep           # leave pod alive on failure

Without --gpu, the orchestrator iterates through CANDIDATE_GPUS and tries each
in order until one create-pod call succeeds. This dodges RunPod's inventory
churn — when a specific card is sold out, the next-best one usually isn't.

The pod is terminated in a finally block, so a crashed eval or a Ctrl+C
won't leave a billable GPU running. Pass --keep to opt out of that safety
net (useful for debugging on the pod after a failure).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

try:
    import runpod
except ImportError:
    raise SystemExit("runpod SDK not installed. Run: uv sync")

from scripts.pod_down import terminate_pod
from scripts.pod_up import (
    DEFAULT_DISK_GB,
    DEFAULT_GPU,
    DEFAULT_IMAGE,
    DEFAULT_UPLOAD,
    POD_NAME,
    REPO_DIR_ON_POD,
    STATE_FILE,
    provision_pod,
)


LOCAL_REPORTS_DIR = Path("runs/eval_reports")
# astral's uv installer puts uv at ~/.local/bin/uv and patches ~/.bashrc, but
# non-interactive ssh shells skip .bashrc — set PATH explicitly each call.
EVAL_COMMAND = (
    f'export PATH="$HOME/.local/bin:$PATH" && '
    f"cd {REPO_DIR_ON_POD} && uv run --no-sync python -m run eval"
)

# Try in order until one survives the create-pod race. Anything ≥ 16 GB VRAM
# fits Qwen3-4B + LoRA in bf16 with room for KV cache. Ordered cheap-first so
# the wallet wins the tie when multiple cards are live.
CANDIDATE_GPUS = (
    "NVIDIA RTX 4000 Ada Generation",  # ~$0.26/hr, 20 GB
    "NVIDIA RTX A5000",                # ~$0.27/hr, 24 GB
    "NVIDIA RTX A4500",                # ~$0.30/hr, 20 GB
    "NVIDIA RTX A4000",                # ~$0.30/hr, 16 GB
    "NVIDIA L4",                       # ~$0.40/hr, 24 GB
    "NVIDIA A40",                      # ~$0.44/hr, 48 GB
    "NVIDIA RTX 6000 Ada Generation",  # ~$0.77/hr, 48 GB
    "NVIDIA L40",                      # ~$0.79/hr, 48 GB
    "NVIDIA L40S",                     # ~$0.86/hr, 48 GB
    "NVIDIA GeForce RTX 4090",         # ~$0.69/hr, 24 GB (when available)
    "NVIDIA RTX A6000",                # ~$0.79/hr, 48 GB
    "NVIDIA GeForce RTX 5090",         # ~$0.99/hr, 32 GB
    "NVIDIA GeForce RTX 3090",         # ~$0.43/hr, 24 GB (rarely listed)
    "NVIDIA H100 PCIe",                # ~$2.89/hr, 80 GB — expensive fallback
)


def _ssh_base(ssh_key: str, port: int) -> list[str]:
    return [
        "ssh", "-i", ssh_key, "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
    ]


def _scp_base(ssh_key: str, port: int) -> list[str]:
    return [
        "scp", "-i", ssh_key, "-P", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
    ]


def _scp_minimal_env(ssh_key: str, host: str, port: int) -> None:
    """SCP a stripped-down .env to the pod so run/eval's load_dotenv finds keys.

    RunPod's pod-creation `env=` dict gets baked into the container but doesn't
    always reach non-interactive SSH sessions. SCP'ing a .env keeps the eval's
    load_dotenv() happy. We deliberately exclude RUNPOD_API_KEY from what
    lands on the pod — the pod has no reason to be able to manage other pods.
    """
    target = f"root@{host}"
    keys = ("HF_TOKEN", "HF_PUSH_REPO", "ANTHROPIC_API_KEY")
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False
        ) as f:
            tmp_path = f.name
            for key in keys:
                value = os.environ.get(key)
                if value:
                    f.write(f"{key}={value}\n")
        print(f"Uploading .env (keys: {', '.join(keys)})")
        subprocess.run(
            _scp_base(ssh_key, port) + [tmp_path, f"{target}:{REPO_DIR_ON_POD}/.env"],
            check=True,
        )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def _run_eval_on_pod(ssh_key: str, host: str, port: int) -> None:
    print("Running eval on pod (~30 min)...")
    target = f"root@{host}"
    cmd = _ssh_base(ssh_key, port) + [target, EVAL_COMMAND]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Eval failed on pod (exit {result.returncode})")


def _scp_reports_back(ssh_key: str, host: str, port: int) -> list[Path]:
    target = f"root@{host}"
    LOCAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Pulling reports → {LOCAL_REPORTS_DIR}/")
    cmd = _scp_base(ssh_key, port) + [
        f"{target}:{REPO_DIR_ON_POD}/runs/eval_reports/model_eval_*",
        f"{LOCAL_REPORTS_DIR}/",
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"scp of reports failed (exit {result.returncode}). "
            "Reports may still exist on the pod — terminate skipped if --keep."
        )
    landed = sorted(LOCAL_REPORTS_DIR.glob("model_eval_*"))
    if not landed:
        raise RuntimeError(
            f"scp returned 0 but no files landed in {LOCAL_REPORTS_DIR}. "
            "Check the pod for output."
        )
    for f in landed:
        print(f"  {f}")
    return landed


def _provision_with_fallback(candidates: tuple[str, ...], **kwargs) -> tuple[str, str, int]:
    last_error: Exception | None = None
    for gpu in candidates:
        print(f"\n--- Trying {gpu} ---")
        try:
            return provision_pod(gpu=gpu, **kwargs)
        except SystemExit:
            raise  # configuration errors (missing SSH key, etc.) propagate
        except Exception as e:
            print(f"  unavailable: {type(e).__name__}: {str(e).splitlines()[0][:90]}")
            last_error = e
    raise RuntimeError(
        f"All {len(candidates)} GPU candidates unavailable. Last error: {last_error}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--gpu",
        default=None,
        help="Force a single GPU type. Omit to walk CANDIDATE_GPUS.",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--name", default=f"{POD_NAME}-eval")
    parser.add_argument("--disk-gb", type=int, default=DEFAULT_DISK_GB)
    parser.add_argument(
        "--ssh-key",
        default=os.path.expanduser("~/.ssh/id_ed25519_arunma"),
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Don't terminate the pod on failure (default: terminate via finally).",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY not set (add to .env)")
    runpod.api_key = api_key

    candidates = (args.gpu,) if args.gpu else CANDIDATE_GPUS

    pod_id: str | None = None
    eval_succeeded = False
    exit_code = 1

    try:
        pod_id, host, port = _provision_with_fallback(
            candidates,
            image=args.image,
            disk_gb=args.disk_gb,
            name=args.name,
            ssh_key=args.ssh_key,
            upload=DEFAULT_UPLOAD,
            skip_setup=False,
            skip_data=False,
        )

        _scp_minimal_env(args.ssh_key, host, port)
        _run_eval_on_pod(args.ssh_key, host, port)
        _scp_reports_back(args.ssh_key, host, port)
        eval_succeeded = True
        exit_code = 0
        print("\nDone. Reports are in runs/eval_reports/.")
        print("Commit with:")
        print("  git add runs/eval_reports && git commit -m 'eval: model eval snapshot'")

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        exit_code = 130

    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}", file=sys.stderr)
        exit_code = 1

    finally:
        # If provision_pod raised after creating the pod (e.g. timed out waiting
        # for SSH), our local pod_id is still None but STATE_FILE has the id —
        # rescue it so the finally still terminates the orphan.
        if pod_id is None and STATE_FILE.exists():
            pod_id = STATE_FILE.read_text().strip() or None
            if pod_id:
                print(f"\nRecovered orphan pod_id from {STATE_FILE}: {pod_id}")
        if pod_id is not None:
            if not eval_succeeded and args.keep:
                print(
                    f"\nPod {pod_id} kept alive (--keep). Terminate manually with:\n"
                    f"  uv run python -m scripts.pod_down",
                    file=sys.stderr,
                )
            else:
                try:
                    print(f"\nTerminating pod {pod_id}")
                    terminate_pod(pod_id)
                    if STATE_FILE.exists():
                        STATE_FILE.unlink()
                    print("Pod terminated.")
                except Exception as e:
                    print(
                        f"WARN: termination failed: {type(e).__name__}: {e}\n"
                        f"Run manually: uv run python -m scripts.pod_down {pod_id}",
                        file=sys.stderr,
                    )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
