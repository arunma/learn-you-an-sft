from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

PERSONA_PROMPT_PATH = REPO_ROOT / "persona_prompt.md"

RUNS_DIR = REPO_ROOT / "runs"
