"""End-to-end: spin up a RunPod GPU, run `run eval`, scp reports back, terminate.

  uv run python -m scripts.eval_on_pod              # defaults: RTX 4090, val.jsonl
  uv run python -m scripts.eval_on_pod --gpu "NVIDIA RTX A6000"
  uv run python -m scripts.eval_on_pod --keep       # leave pod alive on failure

The pod is terminated in a finally block, so a crashed eval or a Ctrl+C
won't leave a billable GPU running. Pass --keep to opt out of that safety
net (useful for debugging on the pod after a failure).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
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
EVAL_COMMAND = f"cd {REPO_DIR_ON_POD} && uv run --no-sync python -m run eval"


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--gpu", default=DEFAULT_GPU)
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

    pod_id: str | None = None
    eval_succeeded = False
    exit_code = 1

    try:
        pod_id, host, port = provision_pod(
            gpu=args.gpu,
            image=args.image,
            disk_gb=args.disk_gb,
            name=args.name,
            ssh_key=args.ssh_key,
            upload=DEFAULT_UPLOAD,
            skip_setup=False,
            skip_data=False,
        )

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
