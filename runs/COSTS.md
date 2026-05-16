# Training run costs

Running ledger. One row per RunPod session — logged immediately on
termination, before doing anything else. See `RUNPOD.md` § 11 for
the discipline.

## How to log

The act of writing the "What you learned" column is the moment you
actually think about what the run showed. Skip it and runs blur
together.

- **Date:** ISO `YYYY-MM-DD`.
- **Pod:** GPU type (e.g., `H100 80G`).
- **Mins:** wall-clock minutes the pod was billable (launch → terminate, not just training).
- **$/hr:** the hourly rate shown on the RunPod listing when you launched.
- **$:** `Mins/60 × $/hr`, rounded to two decimals.
- **Stage:** which lesson / experiment this is (e.g., `Phase A 13K`, `ablation r=8`).
- **What you learned:** one tight sentence on the result + headline takeaway.

## Synthesis (Gemini API, not pod-billed)

Synthesis runs use the Gemini API directly; track separately because
they don't involve a pod.

| Date       | Service          | Tokens (approx) | $     | Stage          | What you learned |
|------------|------------------|------------------|-------|----------------|------------------|
| YYYY-MM-DD | Gemini Flash     |                  | $0.00 | question pool  | - |
| YYYY-MM-DD | Gemini 2.5 Pro   |                  | $0.00 | 15K bulk synth | - |

## H100 training runs

| Date       | Pod          | Mins | $/hr  | $      | Stage              | What you learned |
|------------|--------------|------|-------|--------|--------------------|------------------|
| YYYY-MM-DD | H100 80G     |      |  2.69 | $0.00  | Phase A — TRL+LoRA | - |
| YYYY-MM-DD | H100 80G     |      |  2.69 | $0.00  | Phase B — hand-rolled | - |
| YYYY-MM-DD | H100 80G     |      |  2.69 | $0.00  | ablation r=8       | - |

## Project budget tracker

| Budget item             | Estimated | Actual |
|-------------------------|----------:|-------:|
| Gemini Flash (questions)|     $0.50 |        |
| Gemini Pro (responses)  |  $30–$50  |        |
| Claude Haiku (judge)    |    $10.00 |        |
| H100 (Phase A)          |     $1–3  |        |
| H100 (Phase B)          |     $1–3  |        |
| H100 (ablations 3–4×)   |     $4–6  |        |
| Buffer                  |    $10–15 |        |
| **Total**               | **~$60–90** | |

> Original project budget in HANDOFF was $50-60; revised upward
> after the persona prompt grew to ~1.5K tokens. Worth it for
> quality.

## Worst-case scenarios to remember

- Forgot to terminate a RunPod pod for 24 hours → **+$72**
- Used Sonnet 4.6 instead of Haiku 4.5 for judge → **+$20**
- Re-ran 15K synth because the first run wasn't persisted → **+$30**
- Trained for 30 epochs by accident → **+$5-10**
