#!/bin/bash
# Build a quantized GGUF from a LoRA adapter merged into its base model,
# ready for local inference via LM Studio / llama.cpp / Ollama.
#
# Idempotent: re-running skips already-completed steps.
#
# Usage:
#     bash scripts/build_gguf.sh
#
# Overridable env vars:
#     ADAPTER_ID         HF Hub id of the LoRA adapter      (default: arunma/monty3)
#     BASE_ID            HF Hub id of the base model        (default: Qwen/Qwen3-4B-Instruct-2507)
#     HF_PUSH_REPO_GGUF  Where to upload the GGUFs          (default: same as ADAPTER_ID)
#     QUANT              Quantization preset                (default: Q4_K_M)
#                        Options: Q4_K_M, Q5_K_M, Q6_K, Q8_0, or "none" / "f16" to skip
#     LLAMA_CPP_DIR      Where to clone+build llama.cpp     (default: /opt/llama.cpp)
#     SKIP_PUSH          Set to 1 to skip HF upload         (default: push if HF_TOKEN is set)
#
# Required:
#     HF_TOKEN env var (for private adapter pull + optional push)
#
# Examples:
#     # default — round 3 (arunma/monty3 on Qwen3-4B)
#     bash scripts/build_gguf.sh
#
#     # build for an older round
#     ADAPTER_ID=arunma/monty BASE_ID=Qwen/Qwen2.5-3B-Instruct bash scripts/build_gguf.sh
#
#     # higher-quality quant
#     QUANT=Q8_0 bash scripts/build_gguf.sh
#
#     # build locally, don't push
#     SKIP_PUSH=1 bash scripts/build_gguf.sh

set -euo pipefail

# ----- Config -----
ADAPTER_ID="${ADAPTER_ID:-arunma/monty3}"
BASE_ID="${BASE_ID:-Qwen/Qwen3-4B-Instruct-2507}"
HF_PUSH_REPO_GGUF="${HF_PUSH_REPO_GGUF:-${ADAPTER_ID}}"
QUANT="${QUANT:-Q4_K_M}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/opt/llama.cpp}"
SKIP_PUSH="${SKIP_PUSH:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${REPO_ROOT}/models"
MERGED_DIR="${MODELS_DIR}/monty-merged"
F16_GGUF="${MODELS_DIR}/monty-4b-f16.gguf"
QUANT_LOWER=$(echo "$QUANT" | tr '[:upper:]' '[:lower:]')
QUANT_GGUF="${MODELS_DIR}/monty-4b-${QUANT_LOWER}.gguf"

echo "============================================================"
echo "  build_gguf.sh"
echo "============================================================"
echo "  Adapter:    ${ADAPTER_ID}"
echo "  Base:       ${BASE_ID}"
echo "  Quant:      ${QUANT}"
echo "  llama.cpp:  ${LLAMA_CPP_DIR}"
echo "  Push to:    $([ "$SKIP_PUSH" = "1" ] && echo "(skipped)" || echo "${HF_PUSH_REPO_GGUF}")"
echo "============================================================"
echo

mkdir -p "${MODELS_DIR}"
cd "${REPO_ROOT}"

# ----- 1. Merge LoRA into base (skipped if merged dir already exists) -----
if [ -f "${MERGED_DIR}/model.safetensors" ] || [ -f "${MERGED_DIR}/model.safetensors.index.json" ]; then
    echo "==> [1/5] Skip merge (${MERGED_DIR} already exists)"
else
    echo "==> [1/5] Merging LoRA into base..."
    BASE_ID="${BASE_ID}" ADAPTER_ID="${ADAPTER_ID}" \
        uv run python -m inference.merge_for_gguf
fi
echo

# ----- 2. Set up llama.cpp (skipped if already cloned + built) -----
if [ ! -d "${LLAMA_CPP_DIR}" ]; then
    echo "==> [2/5] Cloning llama.cpp + installing build deps..."
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq
        apt-get install -y -qq cmake build-essential
    fi
    git clone --depth=1 https://github.com/ggml-org/llama.cpp "${LLAMA_CPP_DIR}"
    uv pip install -r "${LLAMA_CPP_DIR}/requirements/requirements-convert_hf_to_gguf.txt"
else
    echo "==> [2/5] llama.cpp already cloned at ${LLAMA_CPP_DIR}"
fi

if [ ! -f "${LLAMA_CPP_DIR}/build/bin/llama-quantize" ]; then
    echo "==> [2/5] Building llama.cpp (one-time, ~5 min)..."
    cd "${LLAMA_CPP_DIR}"
    cmake -B build
    cmake --build build --config Release -j
    cd "${REPO_ROOT}"
else
    echo "==> [2/5] llama.cpp already built"
fi
echo

# ----- 3. Convert HF -> GGUF (f16) -----
if [ -f "${F16_GGUF}" ]; then
    echo "==> [3/5] Skip f16 conversion (${F16_GGUF} already exists)"
else
    echo "==> [3/5] Converting HF -> GGUF (f16)..."
    uv run python "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" "${MERGED_DIR}" \
        --outfile "${F16_GGUF}" \
        --outtype f16
fi
echo

# ----- 4. Quantize -----
if [ "$QUANT" = "none" ] || [ "$QUANT" = "f16" ]; then
    echo "==> [4/5] Skip quantization (QUANT=${QUANT})"
    QUANT_GGUF=""
elif [ -f "${QUANT_GGUF}" ]; then
    echo "==> [4/5] Skip quantize (${QUANT_GGUF} already exists)"
else
    echo "==> [4/5] Quantizing to ${QUANT}..."
    "${LLAMA_CPP_DIR}/build/bin/llama-quantize" \
        "${F16_GGUF}" \
        "${QUANT_GGUF}" \
        "${QUANT}"
fi
echo

# ----- 5. Push to HF Hub -----
if [ "${SKIP_PUSH}" = "1" ]; then
    echo "==> [5/5] Skip HF push (SKIP_PUSH=1)"
elif [ -z "${HF_TOKEN:-}" ]; then
    echo "==> [5/5] Skip HF push (HF_TOKEN not set)"
else
    echo "==> [5/5] Pushing GGUFs to ${HF_PUSH_REPO_GGUF}..."
    PUSH_QUANT="${QUANT_GGUF}" PUSH_F16="${F16_GGUF}" PUSH_REPO="${HF_PUSH_REPO_GGUF}" \
        uv run python -c "
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ['HF_TOKEN'])
repo_id = os.environ['PUSH_REPO']

q = os.environ.get('PUSH_QUANT', '')
f = os.environ.get('PUSH_F16', '')

if q:
    fname = os.path.basename(q)
    print(f'Uploading {fname}...')
    api.upload_file(
        path_or_fileobj=q,
        path_in_repo=fname,
        repo_id=repo_id,
        repo_type='model',
        commit_message=f'add {fname} (quantized GGUF for LM Studio / llama.cpp)',
    )

if f:
    fname = os.path.basename(f)
    print(f'Uploading {fname}...')
    api.upload_file(
        path_or_fileobj=f,
        path_in_repo=fname,
        repo_id=repo_id,
        repo_type='model',
        commit_message=f'add {fname} (unquantized GGUF for archival/re-quantization)',
    )

print(f'Pushed: https://huggingface.co/{repo_id}')
"
fi
echo

# ----- Final summary -----
echo "============================================================"
echo "  DONE"
echo "============================================================"
ls -lah "${MODELS_DIR}/" 2>/dev/null | grep -E 'merged|gguf' || true
echo
echo "Pull from HF on your Mac:"
echo "    mkdir -p ~/.lmstudio/models/${HF_PUSH_REPO_GGUF}"
if [ "$QUANT" != "none" ] && [ "$QUANT" != "f16" ]; then
    echo "    hf download ${HF_PUSH_REPO_GGUF} $(basename "${QUANT_GGUF}") \\"
    echo "      --local-dir ~/.lmstudio/models/${HF_PUSH_REPO_GGUF}"
fi
echo
echo "Then open LM Studio, paste the system prompt from train.py:"
echo "    grep -A3 '^SYSTEM' runs/sft_v1_trl/train.py | head -5"
echo "============================================================"
