# Running Monty in LM Studio

Pipeline for turning the published `arunma/monty` LoRA adapter into a GGUF file that LM Studio (or any llama.cpp-based runner) can load.

LM Studio cannot load PEFT adapters directly — it only loads GGUF files via llama.cpp. So the export has four steps:

1. **Merge** the LoRA into the base model weights
2. **Convert** the merged HF model to GGUF
3. **Quantize** (optional but recommended)
4. **Drop into LM Studio's models folder**

End-to-end takes ~5 minutes for a 0.5B model on an M-series Mac.

---

## 1. Merge LoRA → base

The repo ships a helper script:

```bash
uv run python -m inference.merge_for_gguf
```

This:
- Pulls `Qwen/Qwen2.5-0.5B-Instruct` from HF Hub
- Pulls the `arunma/monty` adapter
- Calls `PeftModel.merge_and_unload()` to fold LoRA deltas into the base weights
- Saves the result to `models/monty-merged/` (HF format: `config.json`, `model.safetensors`, tokenizer files)

Output: a ~1 GB folder you can convert directly with llama.cpp.

---

## 2. Convert HF → GGUF

Clone llama.cpp once, then convert:

```bash
# One-time setup
git clone https://github.com/ggml-org/llama.cpp ~/code/llama.cpp
pip install -r ~/code/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

# Convert (from learn-you-an-sft repo root)
python ~/code/llama.cpp/convert_hf_to_gguf.py models/monty-merged \
  --outfile models/monty-f16.gguf \
  --outtype f16
```

Result: `models/monty-f16.gguf` (~1 GB, full bf16 → f16 precision).

---

## 3. Quantize (recommended)

```bash
# One-time: build llama.cpp's quantize tool
cd ~/code/llama.cpp
cmake -B build && cmake --build build --config Release -j

# Quantize to Q4_K_M (best size/quality tradeoff for small models)
./build/bin/llama-quantize \
  /Users/arunmanivannan/projects/ai/learn-you-an-sft/models/monty-f16.gguf \
  /Users/arunmanivannan/projects/ai/learn-you-an-sft/models/monty-q4_k_m.gguf \
  Q4_K_M
```

Result: `monty-q4_k_m.gguf` (~350 MB). For a 0.5B model the quality gap vs. f16 is tiny.

Common quant flavors:

| Format | Size (0.5B) | When |
|---|---|---|
| `Q4_K_M` | ~350 MB | Default. Best size/quality balance. |
| `Q5_K_M` | ~420 MB | Slightly higher fidelity. |
| `Q8_0` | ~600 MB | Near-lossless; pick when disk isn't tight. |
| `f16` | ~1 GB | Skip quantization entirely. |

---

## 4. Install into LM Studio

LM Studio expects this folder structure on macOS:

```
~/.lmstudio/models/
└── arunma/
    └── monty/
        └── monty-q4_k_m.gguf
```

```bash
mkdir -p ~/.lmstudio/models/arunma/monty
cp models/monty-q4_k_m.gguf ~/.lmstudio/models/arunma/monty/
```

---

## 5. Load and chat

1. Open LM Studio → **My Models** (left sidebar). The `arunma/monty` row should appear; click refresh if not.
2. Switch to **Chat** → top-bar model selector → pick `monty-q4_k_m`.
3. Open the right-hand **System Prompt** panel and paste the *exact* system prompt the model was trained with. Find it as the `SYSTEM` constant in `runs/sft_v1_trl/train.py`. Without the matching system prompt the persona will not reliably fire.
4. Click **Load** and chat.

### Recommended inference settings

| Setting | Value | Why |
|---|---|---|
| Temperature | 0.7 | Standard chat range |
| Top-p | 0.9 | Standard |
| Context length | 1024 – 4096 | Trained on 1024; can extend at inference |
| Chat template | Auto (Qwen2.5) | LM Studio reads this from the GGUF metadata |

---

## Troubleshooting

**LM Studio doesn't list the model.**
Check the folder structure exactly: `~/.lmstudio/models/<author>/<modelname>/<file>.gguf`. The `.gguf` extension must be lowercase. Restart LM Studio if the refresh button doesn't pick it up.

**Persona feels weak or generic.**
Almost always a system-prompt mismatch. Copy the `SYSTEM` string from `runs/sft_v1_trl/train.py` verbatim into LM Studio's system prompt field. A different system prompt is effectively a different model at inference time.

**`convert_hf_to_gguf.py` complains about a missing dependency.**
Run `pip install -r ~/code/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt` inside whatever Python env you're using to run the converter. The script needs `gguf`, `transformers`, `torch`, and a few helpers.

**Apple Silicon performance.**
LM Studio uses Metal automatically. The 0.5B model at Q4_K_M runs at ~60-100 tok/s on M-series chips with no extra config.

---

## Re-running the pipeline after a new training run

If you retrain (new adapter pushed to `arunma/monty` or a different repo):

```bash
# 1. Re-merge
uv run python -m inference.merge_for_gguf

# 2. Re-convert
python ~/code/llama.cpp/convert_hf_to_gguf.py models/monty-merged \
  --outfile models/monty-f16.gguf --outtype f16

# 3. Re-quantize
~/code/llama.cpp/build/bin/llama-quantize \
  models/monty-f16.gguf models/monty-q4_k_m.gguf Q4_K_M

# 4. Replace the file in LM Studio's folder
cp models/monty-q4_k_m.gguf ~/.lmstudio/models/arunma/monty/
```

LM Studio picks up the new file on the next model load (no need to restart).
