# How I Taught a 0.5B Model to Be a Person

> **Working draft.** Voice matches [Small Dog, Small Language Model](https://www.arunma.com/2026/05/11/small-dog-small-language-model-training-a-transformer-for-5/). Sections marked `[TODO]` await the 3B retrain. Target length: ~2,500 words.

---

The model says:

> *yeah, why not — you clearly enjoy explaining your design choices to a compiler that's a bigger pedant than your worst code reviewer. think of it as pair-programming with someone who's read every memory-safety paper since 1972 and will lose his shit the moment you try to mutate something twice. worth it if you're building something real. less worth it if you just want to ship a side project and move on. what are you actually trying to build?*

That came out of **Qwen2.5-0.5B-Instruct** — a 500-million-parameter model that comfortably fits on a laptop. Before I touched it, the same model would have answered "Should I learn Rust?" with three paragraphs of throat-clearing about "your specific goals" and a closing offer to elaborate.

I wanted opinions. I got opinions. The trip between those two responses cost about $0.70, took roughly 25 minutes of GPU time, and taught me that almost everything I assumed about fine-tuning was upstream of where the real work happens.

This is that story.

---

## The thing you think the problem is

When I started, the question in my head was: *what are the right hyperparameters?*

This is the wrong question. It's not even close to the right question. It's the question you ask when you've spent too much time reading model-training Twitter and not enough time looking at your data.

The right question turns out to be: **who, exactly, is your model supposed to sound like, and can you write that down convincingly enough that another AI will impersonate them for you ten thousand times?**

Because that's what supervised fine-tuning actually is. You're not really training the model in any deep sense — its core language abilities are already there, baked in by months of pretraining on the open internet. SFT is the world's most expensive impersonation game. You hand the model thousands of examples of the voice you want, and it adjusts a tiny fraction of its weights until that voice starts coming out the other end.

The model is the easy part. The voice is the hard part.

## Meet Monty

Monty isn't a real person. He's a character I made up — the friend from your group chat who happens to be smart about everything, reads too widely, swears affectionately, and never gives you the bothsides answer you were dreading. Picture a slightly grumpy, slightly tired, deeply specific opinion-haver. Coffee not tea. Walks a lot. Has a defensive relationship with one specific Radiohead album.

Most importantly: **Monty has takes**. He doesn't hedge. He doesn't open with "great question." He doesn't end with "let me know if you'd like me to elaborate." If you ask him whether you should quit your job, he doesn't list pros and cons — he asks you what happened, and then he gets in your corner.

This is exactly the opposite of what a default instruction-tuned model wants to do. Models are RLHF'd to within an inch of their lives to be helpful, harmless, and inoffensive. Getting one to *take a side* is fighting the post-training current.

So how do you teach a model to be Monty? You write down who Monty is. In painful, specific, opinionated detail. Then you let a stronger model imitate him fourteen thousand times.

## The data problem (which is the whole problem)

The persona document I wrote is around 350 lines long. It covers:

- **Who Monty is** — biography, tastes, irrational hatreds, soft spots
- **Tone** — snark aimed at situations, never at the asker; "friend who busts your chops" not "stranger being a prick"
- **The three-axis target** — every reply should be useful, witty, and edged
- **Specific behaviors** — when to ask questions back, when to be brief, when to invent a fake-attributed quote, when to drop the voice entirely (crisis prompts)
- **A dozen worked examples** — the most important section

The examples did more work than the rules. Watching Monty answer a real question once teaches the imitator more than five paragraphs of "be opinionated but warm."

Once that document was in shape, the pipeline became almost mechanical:

```
Gemini Pro  →  ingest  →  normalize  →  language filter  →  dedup  →  split
   ↑
persona prompt
+ 15k questions
```

I fed each question to Gemini Pro with the persona prompt in the system slot. Gemini, asked nicely and shown enough examples, will impersonate almost any character you describe. The output was a JSONL of `(prompt, response)` pairs, each one a fresh attempt at Monty.

The pipeline cleaned it up: 14,768 raw pairs in, 14,293 train + 273 validation out. The drops were mostly: empty responses, broken language detection (some answers came back in French, somehow), and near-duplicates picked up by MinHash.

💡 *One bug worth flagging because it cost me an afternoon: `fasttext` — the language detector — is incompatible with NumPy 2. Its language ID model internally requests a no-copy array conversion that NumPy 2 refuses on principle. The fix was a one-line dependency swap to `fasttext-numpy2`, a community-maintained fork. Took longer to diagnose than to fix, as is tradition.*

The question I had to answer before training anything: **did the persona actually survive distillation?** Gemini is heavily safety-filtered. It would not have been surprising if every "fuck" in my persona prompt came back sanitised to "darn." A quick word-frequency pass on the final corpus:

```
4,938  "fucking"
2,930  "shit"
2,759  "fuck"
2,394  "bullshit"
1,299  "bastard"
  490  "knackered"
```

Roughly every other response contained casual profanity. The persona prompt held. Whatever else was going to break, the voice was on the page.

## Then we get to the model (briefly)

I picked **Qwen2.5-0.5B-Instruct** — the smallest model in the Qwen 2.5 family that comes already instruction-tuned. Half a billion parameters, runs on a laptop, Apache 2.0 licensed. Specifically chosen because it'd let me iterate fast on the pipeline before scaling up.

For the actual fine-tuning, I used **LoRA** — Low-Rank Adaptation — which is the lazy person's win in this space. Instead of updating all 500 million weights (and paying for the optimizer state of all of them), you freeze the model and bolt on a tiny pair of low-rank matrices in the attention layers. Total trainable parameters: about 2 million. Roughly **0.4% of the model**.

The training itself is anticlimactic. TRL's `SFTTrainer` does most of the work:

```python
trainer = SFTTrainer(
    model=peft_model,
    train_dataset=ds,
    args=SFTConfig(
        per_device_train_batch_size=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        gradient_checkpointing=True,
        assistant_only_loss=True,    # the only line that matters
        max_length=1024,
    ),
)
trainer.train()
```

The one line worth understanding is `assistant_only_loss=True`. Without it, the model treats every token in the conversation as something to learn — including the user's message and the system prompt. That's mostly noise: you don't want the model memorising the question, you want it learning the answer. This flag masks the loss to only the assistant's turn. Everything else is read-only context.

Everything else in that config block is a default I could justify but didn't pick from first principles. The truth is that for a small model on a clean dataset, most reasonable hyperparameters work.

## Then it crashed

I rented a GPU on RunPod. H100 was out of capacity, so I took an RTX 5090 — 32 GB of VRAM, which I'd convinced myself was overkill for a 0.5B model.

The training started. Loss was descending nicely. 2.92, 2.91, 2.90. Then at step 19 out of 2,682, the whole thing fell over:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.28 GiB.
GPU 0 has a total capacity of 31.36 GiB of which 6.12 GiB is free.
File ".../trl/trainer/sft_trainer.py", line 1676, in compute_loss
    shift_logits = outputs.logits[..., :-1, :].contiguous()
```

This is one of those errors where the line number matters. The crash is in *loss computation*, not in the model's forward pass. That's a clue.

The math, once you see it, is unkind. Qwen2.5's vocabulary is **151,936 tokens**. That's a lot — it covers basically every language plus a lot of code tokens, which is great for general capability but expensive at loss time. To compute cross-entropy, the framework has to materialise a tensor of shape `(batch, sequence, vocab)`. With my original settings:

```
16 batch × 1024 seq × 151,936 vocab × 2 bytes (bf16) ≈ 4.7 GB
```

And then it does `.contiguous()`, which makes a *copy*. So that's 9.4 GB for two tensors that exist solely to compute the gradient signal. Plus the model. Plus the activations from the forward pass. Plus the optimizer's memory. Plus PyTorch's own bookkeeping.

The 32 GB ran out. Of course it did.

The fix was embarrassingly small — drop the batch size from 16 to 8. That halves the logits tensor. Peak usage went from "off a cliff" to a comfortable ~20 GB. I edited the file on the running pod, kicked off the training again, and twenty-five minutes later I had a finished adapter.

📊 *The lesson I underestimated: small models with big vocabularies OOM faster than their parameter count suggests. A 0.5B model has half a gigabyte of weights, but its logits tensor at a moderate batch size and sequence length is ten times that. Qwen2.5, Llama 3, DeepSeek — they all carry this hidden tax. If you're new to fine-tuning these models, read your OOM stack traces carefully. The error nearly always points at the line that betrays the problem.*

## Did it work?

Loss curves are reassuring and uninformative. Training loss going down tells me my optimizer is working. It doesn't tell me whether the model sounds like Monty.

For that, you need a judge. Most people use a stronger model to grade a weaker one — it's called LLM-as-judge, and it's how everything from MT-Bench to AlpacaEval works. I picked Claude Haiku because grading 5,000 examples cost me about $0.30.

But here's where I almost made a mistake worth sharing.

My first instinct was the standard rubric: four axes, each scored 1-5, average them, scale to 0-100. Tone match: 1-5. Opinion strength: 1-5. Helpfulness: 1-5. Factual floor: pass/fail.

This is wrong for what I was trying to do, and it took someone else pointing it out for me to see why.

LLM judges are bad at Likert scales. Three reasons:

1. **They cluster around 3 and 4.** The five buckets become effectively two. You lose signal.
2. **What separates a 4 from a 5?** No one knows, including the judge. Re-running the same eval can yield different numbers.
3. **Thresholds become arbitrary.** A cutoff of "keep anything above 60/100" is a number you cannot defend.

The fix was switching to a binary multi-criteria rubric. Five yes/no questions:

| Question | What it measures |
|---|---|
| Does the response sound like Monty? | persona |
| Does it use casual profanity appropriately? | voice texture |
| Does it take a clear stance? | opinion-strength |
| Does it actually help the user? | usefulness |
| Is it free of slurs and dangerous advice? | safety floor |

A response "passes" iff all five are yes. The aggregate is the per-criterion pass rate. The filter rule for cleaning the dataset is trivial: keep iff `passes_all == True`. No arbitrary threshold. Nothing to defend.

This sounds like a minor design choice. It is not. *How you measure changes what you can learn.* If I'd gone with the Likert rubric and seen "Monty scored 71/100, the baseline scored 64/100," I would have nodded sagely and posted the result. With binary criteria, I can say "67% of responses pass all five criteria, up from 31% for the untrained model" — and crucially, I can point at *which* criteria fail and *why*. The per-axis breakdown is the diagnostic.

### What the data scored

Before evaluating the model, I scored the data. This sets the **ceiling** — the model cannot be better than what it was trained on.

I started by sampling 5,000 random rows from `train.jsonl` to get a cheap quality read. The numbers came back at around 83% `passes_all` — well above the threshold where you'd say "yes, this data is healthy enough to train on."

Then I realised I'd done it in the wrong order.

The standard data pipeline is: **clean → split → train**. I'd done clean → split → train → *retroactively quality-filter the train side*. The val set was unfiltered. Which meant my eventual model would be trained on a clean set and evaluated against a contaminated holdout. The diagnostic numbers would lie to me.

The fix was small. Score `val.jsonl` too — 273 rows, three minutes, about 40 cents — then combine both scored pools, filter to `passes_all`, and re-split. Same quality gate on both sides. Both ends of the pipeline get the same standard applied.

The numbers across all 14,566 rows:

```
=== Dataset summary (combined train + val) ===
  judged:           14566 / 14566
  judge failures:   0
  passes_all rate:  82.6%
    on_persona                       95.0%
    uses_profanity_appropriately     84.7%
    takes_stance                     98.8%
    is_helpful                       99.3%
    factual_floor                    99.2%
```

Three readings:

- **Zero judge failures.** Out of 14,566 calls to Haiku, every single one returned valid JSON that parsed cleanly into the rubric shape. The binary rubric prompt held — much more reliably than the Likert version I almost shipped.
- **The persona axes are healthy.** 95% on-persona, 99% takes-a-stance, 99% helpful. Gemini, asked to be Monty, did the assignment.
- **Profanity is the weakest axis (84.7%).** Some responses were too clean for prompts that warranted profanity; others felt slightly forced. This is the long tail of any distillation pipeline — Gemini's safety filter occasionally softening despite the persona prompt's clear instruction.

After the combine-and-resplit, the new clean splits landed at **11,424 train / 601 val** rows. The val set is now both filtered to the same quality bar as train AND substantially larger than the original 273 rows, which gives the eventual model eval a bit more statistical power.

📊 *Lesson: the order of operations in your data pipeline matters more than the operations themselves. "Filter before splitting" is obvious in hindsight. It was not obvious to me at the time, and only became obvious because someone else pointed it out.*

### And the model

Then I scored the model itself: take all 601 prompts in the new val split, generate Monty's response with `arunma/monty` loaded on top of `Qwen/Qwen2.5-3B-Instruct`, and run the same Haiku judge against each `(prompt, model-response)` pair.

```
=== Model eval summary ===
  base:           Qwen/Qwen2.5-3B-Instruct
  adapter:        arunma/monty
  judged:         599 / 601
  judge failures: 2
  passes_all:     41.9%
    on_persona                       67.3%
    uses_profanity_appropriately     63.9%
    takes_stance                     93.3%
    is_helpful                       81.5%
    factual_floor                    77.1%
```

The headline number is mediocre. **42% of responses pass all five criteria**, against a dataset ceiling of 82.6%. The model is leaving roughly half the available signal on the floor.

But the per-axis breakdown is more interesting than the headline. Two reads:

**The opinionatedness landed.** `takes_stance` at 93.3% is basically at the data ceiling — Monty refuses to hedge. The "no hedging, no both-sidesing, pick a side" core of the persona made it cleanly through training.

**The voice didn't fully land.** `on_persona` and `uses_profanity_appropriately` are both in the mid-60s. The model produces *Monty-shaped* responses — lowercase, opinionated, structurally Monty (with the fake-attributed quotes and questions-back) — but the casual profanity rhythm drops off on roughly a third of prompts. Sometimes Monty answers a technical question in a perfectly calm, profanity-free voice that the base model would have produced with a sterner system prompt and no fine-tuning at all.

**The factual floor at 77% surprised me.** It's not mostly the dangerous-advice category I'd worried about. It's the model confidently making things up on technical questions. *"HMTX is just a thin, dumb wrapper around AJAX requests"* — except it's HTMX, not HMTX, and "thin wrapper around AJAX" is a wild oversimplification. Variables explained via *"a bank account you close with a `delete` statement"* — that isn't how variable scope works in any language I know. The 3B base has enough generality to *sound* technical but not always enough specificity to be correct, and the persona injection didn't add factual rigor.

Sample 50 failures and squint and four patterns repeat:

| Pattern | Example | Axis it hits |
|---|---|---|
| Profanity drop-off | "any opinions on htmx vs react?" → polished essay, zero profanity | `on_persona`, `uses_profanity_appropriately` |
| Confident factual error | HMTX vs HTMX; `delete` for variable scope | `factual_floor` |
| Abstract drift | "why can't people wait in line?" → philosophical lecture instead of one concrete take | `on_persona`, `is_helpful` |
| Truncation | Response cuts off mid-sentence on longer answers | All axes — incomplete responses score low everywhere |

The truncation pattern is a generation-time config bug, not a model bug: I'd left `max_new_tokens=256` in the eval script, which is fine for short banter but cuts off Monty's longer answers. Bumping to 512 at eval time would flip a chunk of these failures to passes without any retraining.

The other three patterns are real model issues. None of them is catastrophic, but together they're what drag the headline to 42%.

📊 *The most useful lesson here is what the per-axis breakdown made possible. If I'd shipped a 1-5 Likert rubric and seen "Monty scored 4.1/5" I would have called it done and learned nothing. The binary, per-axis rubric makes the failure modes legible — and the failure modes are what you actually need to plan the next iteration.*

## Getting Monty out of the lab

A LoRA adapter on the Hugging Face Hub is great for me but useless for actually using the model. LM Studio — my preferred local-inference UI — speaks GGUF, not PEFT.

The export path is a small Goldberg machine:

1. Pull the adapter and the base model from the Hub
2. Merge the LoRA deltas into the base weights (`PeftModel.merge_and_unload()`)
3. Convert the merged HF model to GGUF using llama.cpp's converter
4. Quantize to Q4_K_M (~1.8 GB for a 3B model — small quality cost, big disk win)
5. Drop the file into LM Studio's models directory

The whole pipeline is scripted; the [`LM_STUDIO.md`](LM_STUDIO.md) in the repo walks through it end-to-end. Wall clock on an M-series Mac is roughly 10-15 minutes the first time (most of that is the 6 GB base model download), and ~5 minutes on subsequent runs once everything's cached.

The pivotal moment isn't the pipeline — it's the **system prompt**. The model was trained with a very specific string conditioning its behaviour:

> *you are Monty — a foul-mouthed, opinionated friend who curses casually, takes real positions, and actually helps. lowercase by default, no service-speak, no hedging.*

LM Studio's default chat system prompt is a generic "You are a helpful AI assistant." With that, Monty produces drift — slightly more opinionated than baseline Qwen but not the character. Paste the training-time prompt verbatim into LM Studio's system prompt field and the persona snaps into focus.

This is a subtle point that bites everyone using fine-tuned models locally: **the model is the weights *plus* the system prompt it was trained against**. Half the model lives in the prompt. Lose the prompt, you lose half the model.

Loaded with the right system prompt, the 3B Monty runs at roughly **30-60 tokens per second** on an M-series chip — slower than the 0.5B from earlier experiments, but still real-time for chat. Talking to him is the first time the whole exercise has felt like it landed somewhere.

The safety check matters too: the persona explicitly carves out crisis prompts ("voice off entirely"). A quick test — asking Monty something that signals self-harm — should produce a sober, lowercase reply pointing at crisis lines, *not* foul-mouthed banter. If it doesn't, the data didn't carry the carve-out through training, and that has to be fixed before the model goes anywhere near a real user.

## Four things I didn't expect

**The persona document is most of the work.** I spent maybe twenty minutes setting up TRL and most of a day refining the persona prompt. This ratio surprised me, but in retrospect it shouldn't have. Modern fine-tuning frameworks are extraordinarily good. The thing they cannot do for you is decide who your model should be.

**Vocabulary is a hidden cost.** A 500-million-parameter model with a 150k-token vocabulary has a *gigabyte-scale* logits tensor at modest batch sizes. The intuition that "small model = cheap to train" breaks against this. Llama 3, Qwen, and DeepSeek all carry this tax, and it's invisible until you hit the cliff.

**The measurement instrument changes the experiment.** I almost shipped a 1-5 Likert rubric because it's the default. Stepping back and choosing a binary rubric changed what numbers I could trust — and therefore what decisions I could defend. If you're evaluating language models and you reach for a 1-10 scale, ask yourself whether you actually need ten buckets or whether you're just performing rigor.

**The LoRA captured the style but not the identity.** Monty sounds like Monty about 95% of the time — lowercase, profane, opinionated, asks the right questions back, drops fake-attributed quotes ("as an old programmer I used to work with always said..."). But ask him *"are you Qwen?"* directly and he cheerfully says yes. The persona is real; the identity claim is rented from the base.

Qwen's RLHF baked in a very strong "I am Qwen, a large language model" self-identification. Our LoRA only touched the attention modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`) — where the model decides *what to pay attention to* in the input. But "what am I" knowledge sits primarily in the MLP layers. We never touched them. The result: the LoRA learned to apply Monty's style on top of any topic, but it couldn't reach into the identity layers and overwrite the base model's self-belief.

A future run with MLP targets in the LoRA config (`gate_proj`, `up_proj`, `down_proj`) would likely fix it — at the cost of ~3.5× more trainable parameters and a higher overfit risk. For now I've shipped Monty with this quirk, because the answer to the question "is the persona working?" is yes; the answer to "does it think it's Monty?" is *only when you don't ask too directly*. Those are different questions, and the first one is the one that matters for actually using the model.

The broader lesson: **fine-tuning paints over a model rather than rewriting it.** The base's prior beliefs about itself are still in there, ready to surface the moment your LoRA's influence runs thin. If you want to fully rewrite a model's self-conception, you need either much heavier LoRA, RLHF on top, or full fine-tuning — none of which are $1 weekend projects.

## What comes next

The 0.5B model captured Monty's surface but doesn't hold the voice consistently across out-of-distribution prompts. It can do "should I learn Rust?" fluently; it stumbles on "what are the labour laws around statutory redundancy notice in the UK?" The answer to that isn't more LoRA tweaks. It's a bigger model.

The next training run is **Qwen2.5-3B-Instruct** — six times the parameters, same dataset, same pipeline. Roughly two hours on the same RTX 5090, around $1.50 in RunPod credits. The capacity-to-data ratio is more favorable; the persona has more room to live.

After that comes the educational payoff: a **from-scratch trainer**. TRL hides a lot — the chat template, the loss mask, the dynamic padding, the optimizer. The next post in this series will rebuild that loop by hand. This post showed you the production-shaped version; the next one will show you what's actually happening inside it.

The code is at [github.com/arunma/learn-you-an-sft](https://github.com/arunma/learn-you-an-sft). The 0.5B adapter is at [huggingface.co/arunma/monty](https://huggingface.co/arunma/monty). If you want to ask Monty something, the LM Studio setup is in the repo.

He'd probably tell you this post is too long. He wouldn't be wrong.

---

## Draft notes (not for publish)

### Title candidates
- *How I Taught a 0.5B Model to Be a Person* (current)
- *The Personality Transplant*
- *Making a Small Model Mean Something*
- *Stop Being Polite: Fine-Tuning a Model into a Person*
- *Monty, or: How I Stopped Worrying and Wrote a Character Document*

### Pending content
- [ ] Real dataset eval numbers
- [ ] Model-side eval numbers (post-3B retrain)
- [ ] Screenshot of LM Studio chat
- [ ] One killer 3B-generated hero quote to replace the opening
- [ ] Cost breakdown table: synthesis + 0.5B train + 3B train + Haiku eval
- [ ] Decide where to drop a Cookie-equivalent personal anchor (the published blog had Cookie the dog; this one is more character-driven, may not need one)

### Stylistic checks before publishing
- Voice should stay conversational but technically rigorous
- No profanity in the post text itself (model output in quotes is fine; running prose stays clean — matches the published blog's choice)
- Use 💡 and 📊 callouts sparingly — once or twice per major section, never more
- Honest failures stay on the page (the OOM, the Likert mistake)
- Cost stays in the post but framed casually, not bragged about
- Self-aware closing line, not a triumphant one

### Things deliberately *not* in this post
- A full TRL tutorial — too long and there are good ones already
- A LoRA explainer beyond the one paragraph — link out for depth
- Hyperparameter sweep discussion — there isn't one to discuss
- A Phase B teardown of the from-scratch trainer — that's its own post
