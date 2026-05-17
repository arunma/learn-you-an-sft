"""Terminate a RunPod pod.

Reads .env for RUNPOD_API_KEY.

Usage:
  uv run python -m scripts.pod_down                  # terminate pod from .runpod_pod_id
  uv run python -m scripts.pod_down <pod_id>         # terminate a specific pod

Fallback if the SDK call fails (e.g. RUNPOD_API_KEY not set):
  runpodctl remove pod <pod_id>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    import runpod
except ImportError:
    raise SystemExit("runpod SDK not installed. Run: uv add runpod")


STATE_FILE = Path(".runpod_pod_id")


def main() -> int:
    load_dotenv()
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit(
            "RUNPOD_API_KEY not set (add to .env or export).\n"
            "Or use the CLI directly: runpodctl remove pod <pod_id>"
        )
    runpod.api_key = api_key

    if len(sys.argv) > 1:
        pod_id = sys.argv[1]
        source = "command-line arg"
    elif STATE_FILE.exists():
        pod_id = STATE_FILE.read_text().strip()
        source = f"{STATE_FILE}"
    else:
        raise SystemExit(
            "No pod_id given and no .runpod_pod_id state file found.\n"
            "Usage: uv run python -m scripts.pod_down <pod_id>"
        )

    if not pod_id:
        raise SystemExit(f"Empty pod_id from {source}")

    print(f"Terminating pod: {pod_id}  (from {source})")
    try:
        runpod.terminate_pod(pod_id)
    except Exception as e:
        raise SystemExit(
            f"Failed to terminate via SDK: {type(e).__name__}: {e}\n"
            f"Fallback: runpodctl remove pod {pod_id}"
        )

    print("Terminated.")
    if STATE_FILE.exists():
        STATE_FILE.unlink()
        print(f"Cleared {STATE_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
