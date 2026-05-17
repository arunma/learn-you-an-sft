# RunPod — Training on H100 (operations playbook)

A step-by-step guide for actually running training on a cloud H100,
not the conceptual content from [TUTORIAL.md](TUTORIAL.md). Goes
from *"I have a working training script locally"* to *"the adapter
is back on my Mac, the pod is terminated, and the cost is logged."*

This is the operator's manual. Read it once end-to-end before
launching anything.

---

## Table of contents

1. [When to use RunPod (vs Mac local)](#1-when-to-use-runpod-vs-mac-local)
2. [Pre-flight checklist — do BEFORE clicking deploy](#2-pre-flight-checklist--do-before-clicking-deploy)
3. [Launching a pod](#3-launching-a-pod)
4. [Connecting (SSH)](#4-connecting-ssh)
5. [Code + environment setup](#5-code--environment-setup)
6. [Getting the training data onto the pod](#6-getting-the-training-data-onto-the-pod)
7. [Adjusting training config for H100](#7-adjusting-training-config-for-h100)
8. [Running training under tmux](#8-running-training-under-tmux)
9. [Pulling the checkpoint back](#9-pulling-the-checkpoint-back)
10. [**Terminating the pod**](#10-terminating-the-pod-critical)
11. [Cost ledger discipline](#11-cost-ledger-discipline)
12. [Failure modes](#12-failure-modes)
13. [End-to-end cheat sheet](#13-end-to-end-cheat-sheet)

---

## 1. When to use RunPod (vs Mac local)

| Scenario | Where to run | Why |
|---|---|---|
| Lesson 5's 24-example smoke test | **Mac local** | 2–3 min on CPU. Free. |
| Lesson 9 synthesis (Gemini API calls) | **Mac local** | No GPU needed. API-bound, not compute-bound. |
| Lesson 2b filter pipeline | **Mac local** | Few minutes on CPU. fasttext model is the bottleneck. |
| Real training run on 15K examples | **RunPod H100** | Mac CPU would take hours; MPS is uneven. H100 = 10–20 minutes. |
| Phase B hand-rolled trainer on 15K | **RunPod H100** | Same as above. |
| Ablation runs (3–4 short experiments) | **RunPod H100** | Each ~10 min. Cheaper to batch them in one pod session than relaunch. |

**Crossover heuristic:** if the training run takes > 30 minutes on
Mac, it's worth $2 of H100 time to get it done in under 15.

---

## 2. Pre-flight checklist — do BEFORE clicking deploy

The single most expensive mistake is to launch a pod and discover
you forgot something. The H100 meter starts the second the pod is
running, even if you're just sitting at the prompt thinking.

- [ ] **Lesson 9 done.** `data/processed/train.jsonl` exists, has
  ~15K rows. Verify with `wc -l data/processed/train.jsonl`.
- [ ] **manifest.json exists** at `data/processed/manifest.json`,
  reproducibility hashes recorded.
- [ ] **Code is on GitHub `main`.** Don't fix bugs on the pod — fix
  locally, push, then `git pull` on the pod.
  ```bash
  git status                # should show nothing
  git log -1 --oneline      # should match origin/main
  git ls-remote origin main # should match local HEAD
  ```
- [ ] **RunPod account funded.** ~$10 buys ~3–4 hours of H100. The
  full Phase A + Phase B + ablations campaign should fit under $20.
- [ ] **Calendar reminder set** for 1 hour after planned launch.
  Single most important habit; see § 10.
- [ ] **`runs/COSTS.md` ready** to log this session.
- [ ] **Estimate the run time** so you know what "too long" looks
  like. If your training-step estimate says 12 minutes and you're
  at 40 minutes, something is wrong — don't let it silently grind.

If any of these isn't done, **don't launch the pod yet**. Step away,
finish prep, come back.

---

## 3. Launching a pod

### Pick the GPU

**Memory math for our use case** (Qwen2.5-0.5B + LoRA, batch 16,
seq 1024, bf16, gradient_checkpointing):

| Component | VRAM |
|---|---:|
| Base model weights (0.5B, bf16) | ~1 GB |
| LoRA adapter (rank 16, attention-only) | ~10 MB |
| Optimizer state (AdamW on LoRA only) | ~40 MB |
| Activations (with gradient checkpointing) | ~3–6 GB |
| **Total** | **~6–10 GB** |

**Anything ≥ 16 GB VRAM works.** ≥ 24 GB is comfortable. Don't pay
for H100 when you don't need to.

**Picker priority (in order of preference for this project):**

| GPU | $/hr | VRAM | Est. run time | Est. our cost | Notes |
|---|---:|---:|---|---:|---|
| **RTX 5090** | **$0.99** | **32 GB** | **~25 min** | **~$0.40** | Best pick when H100 is out — Medium availability, newest, fast bf16 |
| RTX 4090 | $0.69 | 24 GB | ~25-30 min | ~$0.30 | Cheap, well-tested for ML |
| A100 SXM | $1.49 | 80 GB | ~15-18 min | ~$0.40 | Data-center-grade; classic ML choice |
| RTX 6000 Ada | $0.77 | 48 GB | ~20-25 min | ~$0.30 | Mid-range data-center |
| H100 NVL / SXM | $2.99–$3.07 | 80–94 GB | ~13-15 min | ~$0.70 | Fastest; often out of capacity |

**What to avoid for this project:**

- **L4** (24 GB, $0.39/hr) — cheap but lower memory bandwidth; ~1.5×
  slower than 4090 for training.
- **B-series, H200** — way overkill; only available at top tier; often
  out of capacity anyway.
- **Anything < 16 GB** — tight margin; not worth the savings.

### H100 out of capacity? (the common case)

H100 is in heavy demand and frequently shows "Out of capacity" on
community-cloud. **Don't wait for it.** Pick **RTX 5090** if
available (Medium > Low for likelihood of actual launch), otherwise
fall back through the picker order above. Everything in this
playbook works identically on any of those GPUs — same SSH flow,
same `train.py`, same commands. Only the line you write in
`runs/COSTS.md` changes (`RTX 5090` instead of `H100 80G`).

For the original H100 cost estimates in this doc (e.g., `$2.69/hr`
in § 11), substitute whatever rate you actually launched at. Real
spend will be lower with the alternatives.

### Template

Choose the **PyTorch** template. It comes with:
- Python 3.10 or 3.11
- CUDA + PyTorch pre-installed
- SSH access enabled
- ~50 GB persistent volume

Other templates (TensorFlow, raw Ubuntu) work but require more
setup.

### Deploy

1. Click **Deploy** on the chosen GPU.
2. Confirm:
   - GPU: H100 80GB (or your chosen tier)
   - Disk: ≥ 30 GB (default fine)
   - Template: PyTorch
   - Region: closest to you for SSH latency (US-East, EU-Central, etc.)
3. Wait for the pod status to show **Running** (~30 seconds).
4. Copy the **SSH connection string** from the pod's "Connect" tab.

It usually looks like:

```
ssh root@<host> -p <port> -i ~/.ssh/id_ed25519
```

---

## 4. Connecting (SSH)

If RunPod doesn't already have your SSH public key on file, add it
first: **Settings → SSH keys → paste contents of `~/.ssh/id_ed25519.pub`**.
(Or use your `id_ed25519_arunma.pub` if you keep keys per-identity.)

Then from your Mac:

```bash
ssh root@<host> -p <port>
```

First-time host-key prompt — say yes. You should land at
`root@<container-id>:~#`.

Quick sanity checks:

```bash
nvidia-smi             # confirm GPU is visible
python --version       # 3.10+
df -h /                # check disk
```

---

## 5. Code + environment setup

```bash
# Clone the repo (HTTPS works for public repos; no auth needed)
cd /workspace
git clone https://github.com/arunma/learn-you-an-sft.git
cd learn-you-an-sft

# Install uv if not present (it's faster than pip; the PyTorch
# template usually doesn't ship it)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create venv + install everything from pyproject.toml
uv venv
uv pip install -e .

# bitsandbytes is in pyproject only for Linux — verify it installed
python -c "import bitsandbytes; print(bitsandbytes.__version__)"
```

That last line is worth running. `bitsandbytes` is the 8-bit
optimizer + quantization library; macOS doesn't have it (per the
PEP 508 marker in `pyproject.toml`). On Linux/CUDA you want it.

---

## 6. Getting the training data onto the pod

`data/processed/*.jsonl` is gitignored — for good reason; it's
regenerable and large. Three options for getting it onto the pod:

### Option A: `scp` from your Mac (simplest for a single run)

From your **Mac** (in a separate terminal, not the SSH session):

```bash
cd /Users/arunmanivannan/projects/ai/learn-you-an-sft

scp -P <pod-port> \
  data/processed/train.jsonl \
  data/processed/val.jsonl \
  data/processed/manifest.json \
  root@<host>:/workspace/learn-you-an-sft/data/processed/
```

Pros: no cloud storage needed. Cons: ties this pod to your laptop;
not reproducible if you launch another pod later.

### Option B: Push to HuggingFace Hub (reproducible)

On the **Mac**:

```bash
huggingface-cli login   # one-time, paste your HF token

python -c "
from datasets import Dataset
import json
rows = [json.loads(l) for l in open('data/processed/train.jsonl')]
Dataset.from_list(rows).push_to_hub('arunma/learn-you-an-sft-train', private=True)
"
```

On the **pod**:

```bash
huggingface-cli login   # paste same token
python -c "
from datasets import load_dataset
ds = load_dataset('arunma/learn-you-an-sft-train', split='train')
ds.to_json('data/processed/train.jsonl')
"
```

Pros: reproducible — any future pod gets identical data. Cons:
extra setup; private dataset on HF.

### Option C: Generate synth ON the pod

```bash
export GEMINI_API_KEY=...
uv run python -m synthesis.generate --count 15000
uv run python -m data.filter.pipeline
```

Pros: zero local prep. Cons: you're paying GPU rates while Gemini
API calls run (which is CPU-bound). **Don't do this for cost
reasons.** Always do data prep on the cheapest compute available
(your Mac).

**Recommendation: Option A for first run, Option B for repeatable
training.**

---

## 7. Adjusting training config for H100

Open `runs/sft_v1_trl/train.py` and flip a few knobs that were
conservative for Mac portability:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,           # was torch.float32 on Mac
)

# ... and in SFTConfig:

config = SFTConfig(
    ...,
    per_device_train_batch_size=16,    # was 4 on Mac; H100 has headroom
    gradient_accumulation_steps=1,     # explicit; bump if VRAM gets tight
    bf16=True,                          # was False on Mac
    fp16=False,
    gradient_checkpointing=True,        # save VRAM, ~10% slowdown
    ...
)

# Drop EPOCHS now that we have 15K examples instead of 24:
EPOCHS = 3       # was 10 — at 15K examples, 3 epochs is plenty
```

Mental math for the new run:
- 15,000 examples ÷ 16 batch_size = **938 steps/epoch**
- 3 epochs × 938 = **2,814 total steps**
- H100 step time: ~250–400 ms for 0.5B + LoRA in bf16
- Expected runtime: **12–18 minutes**

**Commit and push these changes from your Mac** before pulling on the
pod — keep the pod in "just run code, don't author" mode:

```bash
# On Mac
git add runs/sft_v1_trl/train.py
git commit -m "config: H100 settings (bf16, bs=16, 3 epochs)"
git push

# On pod
git pull
```

---

## 8. Running training under tmux

**Always** use `tmux`. SSH connections drop. Without `tmux`, a
dropped connection kills the training process and you've burnt N
minutes of H100 time on nothing.

```bash
# Inside the SSH session on the pod
tmux new -s train

# Now you're in a tmux session; start training
uv run python -m runs.sft_v1_trl.train 2>&1 | tee runs/sft_v1_trl/train.log
```

The `2>&1 | tee` captures stdout/stderr to a log file too — useful
for retroactive debugging.

**Detach** from tmux: `Ctrl-b` then `d`. The training keeps running
even if your SSH disconnects.

**Re-attach** (after reconnecting via SSH):

```bash
tmux attach -t train
```

**Watch GPU utilization** in a separate tmux window (`Ctrl-b c`):

```bash
watch -n 1 nvidia-smi
```

You want to see ~80–95% utilization during training. If it's
sitting at 5%, something is wrong (CPU bottleneck — usually the data
loader; bump `dataloader_num_workers` in SFTConfig).

### Live TensorBoard via SSH port forwarding (optional)

Training script already writes tfevents to
`runs/sft_v1_trl/checkpoints/runs/<timestamp>/` (because
`report_to="tensorboard"` is set in `SFTConfig`). To watch the loss
curve live in your browser:

**1. On the pod (in a new tmux window — `Ctrl-b c`):**

```bash
# Wait until training has produced at least one event file
ls runs/sft_v1_trl/checkpoints/runs/*/   # should show events.out.tfevents.*

# Launch TensorBoard. --bind_all is critical for SSH-tunneled access.
uv pip install tensorboard   # if not already in the venv
tensorboard --logdir runs/sft_v1_trl/checkpoints/runs --port 6006 --bind_all
```

Leave that running. It binds to `0.0.0.0:6006` inside the pod.

**2. On your Mac (in a separate terminal, leave it open):**

```bash
# SSH local-forward: localhost:6006 (Mac) <-> localhost:6006 (pod)
ssh -L 6006:localhost:6006 -p <pod-port> root@<pod-host> -N
```

- `-L 6006:localhost:6006` forwards your Mac's port 6006 to the pod's port 6006.
- `-N` means "don't run a remote command, just keep the tunnel open."
- Leave this terminal open the whole time you want TB access; closing it tears down the tunnel.

**3. Open `http://localhost:6006` in your Mac browser.**

You'll see Monty's training loss descend in real time, plus
`learning_rate`, `grad_norm`, `mean_token_accuracy`, `entropy`. The
plots refresh every ~30 s.

### Alternative: post-hoc TensorBoard (simpler)

If you don't need live monitoring, skip the SSH tunnel. After the
run, `scp` the events files back and run TB on Mac:

```bash
# On Mac, after scp'ing the checkpoint back:
tensorboard --logdir runs/sft_v1_trl/checkpoints/runs
# Open http://localhost:6006
```

For a 15-25 min run, post-hoc is plenty. Live is for longer runs
where you want to kill early if loss diverges.

---

## 9. Pulling the checkpoint back

When training completes, the adapter is at
`runs/sft_v1_trl/checkpoints/final/` on the pod (~10 MB for our
LoRA).

From your **Mac**:

```bash
mkdir -p runs/sft_v1_trl/checkpoints
scp -P <pod-port> -r \
  root@<host>:/workspace/learn-you-an-sft/runs/sft_v1_trl/checkpoints/final \
  runs/sft_v1_trl/checkpoints/
```

Also pull the training log:

```bash
scp -P <pod-port> \
  root@<host>:/workspace/learn-you-an-sft/runs/sft_v1_trl/train.log \
  runs/sft_v1_trl/
```

Verify it loaded:

```bash
ls -la runs/sft_v1_trl/checkpoints/final/
# adapter_config.json, adapter_model.safetensors, tokenizer files, etc.
```

You can also sanity-load on the Mac:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "runs/sft_v1_trl/checkpoints/final")
print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
```

---

## 10. Terminating the pod (critical)

This is the single most important step.

**A forgotten H100 pod running for 24 hours = $72.** A forgotten pod
running for a week = $500. People have done this. Don't.

### Recommended: auto-terminate from the training script

`runs/sft_v1_trl/train.py` supports two opt-in env vars that
turn the pod into a self-cleaning oven. **Use them.**

| Env var | Behaviour |
|---|---|
| `HF_PUSH_REPO=arunma/monty` | After training, push the adapter + tokenizer to a private HF Hub repo. Lets you recover the model without scp'ing — adapter survives pod death. Requires `HF_TOKEN` env var. |
| `TERMINATE_POD_AFTER_TRAIN=1` | After training (and HF push if requested), invoke `runpodctl remove pod $RUNPOD_POD_ID`. Permanently kills the pod (compute + storage gone, billing stops). 30-second countdown gives you time to Ctrl+C if you're watching. |

**Full safety-net launch command** (on the pod, inside tmux):

```bash
# Set the safety env vars. Adjust HF_PUSH_REPO to your namespace.
export HF_TOKEN=hf_...                                   # from huggingface.co/settings/tokens
export HF_PUSH_REPO=arunma/monty
export TERMINATE_POD_AFTER_TRAIN=1

# Train. Pod self-destructs ~30s after the script's final print.
uv run python -m runs.sft_v1_trl.train 2>&1 | tee runs/sft_v1_trl/train.log
```

When `trainer.train()` returns, the script:
1. Saves the adapter locally to `runs/sft_v1_trl/checkpoints/final/`.
2. Runs the sanity-check generation.
3. **Pushes** adapter + tokenizer to `huggingface.co/<HF_PUSH_REPO>` (private).
4. Prints `!! AUTO-TERMINATING POD <id> IN 30 SECONDS !!` and counts down.
5. Calls `runpodctl remove pod <id>`. Pod gone, billing stops.

If the HF push fails, the script **does not terminate the pod** —
fail-safe so you can investigate / scp manually. If you want a
truly belt-and-braces setup, also `scp` the checkpoint to your Mac
*before* the auto-terminate fires (use `tee` to see when training
finishes; you have 30 seconds + the HF push window).

The killer feature: **even if you fall asleep at your laptop, the
pod terminates itself**. Same $50 mistake costs $0 with this setup.

### Manual fallback: the discipline

1. The moment training finishes and your checkpoints are scp'd:
   ```bash
   # On pod (good housekeeping, not strictly required since terminate kills everything)
   exit
   ```

2. Go to the **RunPod web UI** dashboard.

3. Find your pod. Click the trash icon. Confirm.

4. **Refresh the page.** Verify the pod is gone from "Running" and
   "My pods" shows zero active.

5. Open `runs/COSTS.md` and log this session immediately (§ 11).

### Why "via the web UI" and not "via the CLI"

The RunPod CLI's terminate command has occasionally had bugs where
it returns success but the pod keeps running. Always confirm via
the dashboard's running-pods list. The number you want there is
**zero**.

### Calendar reminders save money

Set a calendar reminder for 1h after pod launch. If the meter is
still running when it fires and you didn't expect that, something
is wrong — investigate immediately.

---

## 11. Cost ledger discipline

Maintain `runs/COSTS.md` as a running table. One row per session.
Logged immediately on termination, before you do anything else.

```markdown
| Date       | Pod   | Mins | $/hr | $     | Stage         | What you learned |
|------------|-------|------|------|-------|---------------|------------------|
| 2026-05-17 | H100  |   18 | 2.69 |  0.81 | Phase A 15K   | Loss 0.42 final; sanity check on-persona |
| 2026-05-17 | H100  |   22 | 2.69 |  0.99 | Phase B 15K   | Hand-rolled v2 matches v1 within 5% |
| 2026-05-18 | H100  |   12 | 2.69 |  0.54 | Ablation r=8  | Smaller adapter; persona still transfers |
```

Cost-tracking discipline forces you to *notice* when something has
been running longer than expected. The act of writing the
"What you learned" column is also the moment you actually think
about what the run showed. Skip it and runs blur together.

---

## 12. Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `nvidia-smi` shows no GPU | Wrong template, GPU not attached | Terminate, redeploy with the correct template/GPU |
| `pip install` hangs forever | Pod's region has bad PyPI connection | Switch region (terminate + redeploy) or use `uv pip install` which has better mirroring |
| `git clone` fails with auth error | You tried SSH on a fresh pod without forwarding agent | Use HTTPS clone (public repo) or upload key |
| Training stuck at "0/N steps" for > 2 min | First-step compile (`torch.compile`) — actually normal | Wait 60–90 s; if it's > 3 min, kill and check logs |
| `OutOfMemoryError` step 0 | `BATCH_SIZE` too high for the model | Drop to 8, enable gradient_checkpointing |
| `OutOfMemoryError` step 100+ | KV-cache or accumulation growing; bug | Lower `max_length`; check `gradient_accumulation_steps` |
| GPU sitting at 5% utilization | CPU bottleneck (data loader) | Bump `dataloader_num_workers` to 4–8 |
| Loss NaN after a few hundred steps | bf16 instability with this LR | Drop LR by 5× or fall back to fp32 |
| `tmux` session lost on reconnect | Wrong session name on `tmux attach` | Use named sessions; check `tmux ls` |
| Final checkpoint missing | `save_strategy="no"` somehow set, or pod terminated mid-save | Always `save_strategy="epoch"` + `save_total_limit=1`; never terminate without verifying `runs/.../final/` exists |
| Sanity-check generation on pod errors out | Same Mac issues — `BatchEncoding` / device mismatch | Same fixes; or skip sanity-check on pod and run it locally after `scp` |
| Adapter loaded on Mac but generates gibberish | Tokenizer revision mismatch between pod and Mac | Make sure both used the same `transformers` version (the `uv.lock` file pins this) |

### Things that aren't bugs but look like bugs

- **First H100 step takes 60+ seconds** — `torch.compile` tracing.
- **Loss is high (3.0+) for the first few hundred steps** — model
  has to "find" the persona; this is normal early-training behaviour.
- **GPU util drops to 0% during eval steps** — eval doesn't use the
  forward pass the same way; brief drops are fine.

---

## 13. End-to-end cheat sheet

The fast version. For a complete Phase A run with all safety nets
on, the full sequence from "ready to launch" to "adapter on HF Hub
+ pod auto-terminated."

### Pre-flight (on Mac, before launching anything)

```bash
git log -1 --oneline                  # confirm you're on the latest commit
git push                              # make sure pod can pull latest
wc -l data/processed/train.jsonl      # 14293
wc -l data/processed/val.jsonl        # 273
```

```
[ ] Phone alarm set for now + 90 min     (in case auto-terminate doesn't fire)
[ ] RunPod account funded ($10+ buys this whole project)
[ ] HF_TOKEN copied to clipboard or in local .env
[ ] RTX 5090 selected on RunPod (or whichever GPU is available)
```

### On the pod (after SSH in)

```bash
# === Setup ===
cd /workspace
git clone https://github.com/arunma/learn-you-an-sft.git
cd learn-you-an-sft
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
uv venv && uv pip install -e .
nvidia-smi                            # confirm GPU
```

### Get training data onto the pod (from Mac, separate terminal)

```bash
scp -P <pod-port> \
  data/processed/train.jsonl \
  data/processed/val.jsonl \
  data/processed/manifest.json \
  root@<pod-host>:/workspace/learn-you-an-sft/data/processed/
```

Optionally also `scp .env` so you don't have to type `HF_TOKEN`:

```bash
scp -P <pod-port> .env root@<pod-host>:/workspace/learn-you-an-sft/.env
```

### Train (back on pod, inside tmux)

```bash
tmux new -s train

# Three env vars: HF_TOKEN unlocks Hub push, HF_PUSH_REPO names the
# destination, TERMINATE_POD_AFTER_TRAIN=1 makes the pod self-destruct.
export HF_TOKEN=hf_xxxxxxxxxxxxx              # OR: source .env if you scp'd it
export HF_PUSH_REPO=arunma/monty
export TERMINATE_POD_AFTER_TRAIN=1

# Sanity: RUNPOD_POD_ID should print a UUID
echo "POD_ID = $RUNPOD_POD_ID"

# Launch. Pod will train ~25 min, push adapter to HF Hub, then
# auto-terminate after a 30-second countdown.
uv run python -m runs.sft_v1_trl.train 2>&1 | tee runs/sft_v1_trl/train.log
# Ctrl-b d to detach. tmux attach -t train to reattach.
```

### Optional: live TensorBoard (Mac browser → pod)

```bash
# On pod, separate tmux window (Ctrl-b c):
tensorboard --logdir runs/sft_v1_trl/checkpoints/runs --port 6006 --bind_all

# On Mac, separate terminal — leave it open:
ssh -L 6006:localhost:6006 -p <pod-port> root@<pod-host> -N

# Mac browser:
open http://localhost:6006
```

### When training finishes

The pod auto-terminates ~30 s after the final log line. The adapter
is on HF Hub at `huggingface.co/arunma/monty`. Pull to Mac:

```bash
# Either with the HF CLI…
huggingface-cli download arunma/monty --local-dir runs/sft_v1_trl/checkpoints/final

# …or in Python:
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
model = PeftModel.from_pretrained(base, 'arunma/monty')
print(model.print_trainable_parameters())
"
```

### Final house-keeping (back on Mac)

```bash
# 1. Confirm the pod is gone via the RunPod dashboard
#    "My pods" should show zero running.

# 2. Log the run in runs/COSTS.md
#    Date, GPU, mins, $/hr, $, stage, one line on what you learned.

# 3. Optional: tar the local checkpoint to keep a backup
tar -czf runs/sft_v1_trl/checkpoints/final.tar.gz \
        runs/sft_v1_trl/checkpoints/final
```

### If auto-terminate didn't fire (the safety nets failed)

You'll find out from the phone alarm. Open the RunPod dashboard,
click the trash icon, confirm zero running pods. Update
`runs/COSTS.md` with the actual minutes the pod ran. Then read the
`runs/sft_v1_trl/train.log` to see why the script crashed before
reaching the auto-terminate block.

---

## Final note

The hardest part of cloud GPU work isn't the training — it's the
discipline around the meter. Calendar reminders, cost ledger,
"terminate via web UI" muscle memory, never editing code on the
pod. Build those habits on the first cheap run and you'll be
relaxed about every run after.

The training itself is just the script you already have, running on
a different machine.
