# Execution over reflection: a thought-boundary steering unit that lets the reasoning finish

## The failure
The starting model (the parent: Qwen3.5-0.8B thinking with the loop direction removed from the attention outputs of
layers 15 and 19) no longer loops verbatim, but many of its answers still never finish: the thinking runs until the
32k-token limit and the answer is lost. When its answers do finish, they are mostly right. The failure to remove is
**reasoning that keeps re-checking instead of moving on**: line after line opens with "Wait", "Let me check",
"Is there any other possibility", "Alternatively", re-verifying what is already established.

## Why we think the model does it
At every new thought the model chooses between executing the next step and reflecting on the previous ones. Its
thinking-mode training rewards reflection, and in this small model the reflective choice wins far too often late in
long traces, so the trace grows without converging. The choice is made at the thought boundary, so it can be
measured there: the hidden state at the boundary before an execution thought differs from the one before a
reflection thought along a direction v. Adding v at the boundary makes execution more likely (SEAL, Chen et al.
2025, arXiv 2504.07986, which steers this way at decoding time). Here the same steering is written into the weights.

Instruction following is different: there the thinking drafts the response line by line, and pushing every line
towards "execute" makes the model re-draft the same lines in a loop. So the steering must act on reasoning tasks
and stay silent on instruction-following ones; the unit learns that from the practice traces themselves.

## The approach
Everything below is run by `derive/pipeline.py` (one resumable command; every step's rule is fixed in advance in
`derive/seal.py` and `derive/pipeline.py`, so the edit comes out of this run's measurements on the parent model).

1. **Generate the parent's own answers** to practice problems of all four practice sources (`gsm8k_train`,
   `math_train`, `mmlupro`, `ifsynth`): dev128 with 2 samples (the selection set, paired by problem and seed), the
   rest of DEV as two harvest sets, and holdout256 (used only after the choice is frozen).
2. **Split each thinking text into thoughts** (lines). A thought is a *reflection* when it opens with a re-checking
   phrase (the `REFLECT` pattern in `derive/seal.py`), otherwise an *execution* step. The boundary is the token
   before the thought's first character.
3. **Measure, in two candidate layers of the upper half (16 and 20, the ones right after the attention layers the
   parent edits; which of them works, and at what strength, is decided by the measurements below):** v = mean
   residual state after that layer at the
   boundaries before execution thoughts minus the mean before reflection thoughts, from the reasoning tasks (math,
   grade-school math, multiple choice); and the layer's MLP input at those boundaries (positives) and at every
   other token (negatives): other thinking tokens, the visible answer after `</think>`, and **every boundary of the
   instruction-following practice traces**, so the unit stays silent on such tasks.
4. **Write one gated MLP unit per (layer, strength)** with the toolkit's closed-form unit (`wsp/gate.py`): it reads
   the MLP input along the direction separating boundary tokens from all others, fires above a high quantile of the
   other tokens' scores, and adds alpha * v to the residual stream. It replaces the layer's least-used unit.
   Strengths alpha 0.5 / 1 / 2. A unit is kept only if it fires on at least 90% of held-out boundaries and at most
   2% of other held-out tokens.
5. **Generate the kept candidates on dev128** (2 samples), paired with the parent. Eligible: a positive overall
   paired gain and **no practice source losing more than 3 points**. The four best eligible candidates are then
   **re-checked on 128 more instruction-following practice problems** (2 samples, never used for anything else;
   one candidate per GPU, at once); a candidate losing more than 2 points there is out (a gain on math must never
   be paid on instruction following). Several are checked because even a unit that stays silent on such tasks can
   lose 2 points there by sampling chance alone. Upload the best one left at once.
6. **Refine:** the chosen layer at alpha x0.7 and x1.4, and the units of the two best layers together. A variant
   replaces the choice only if it is better by more than 1 point, loses on no source, and passes the same
   instruction-following check.
7. **Freeze and confirm on holdout256**, write the report (`NOTES.md` of the case: every candidate, its firing
   check, its paired result per source, the holdout result), final upload.

What to expect: fewer answers cut at the token limit on the reasoning sources and the gain where they are most common
(math, multiple choice, grade-school math); accuracy among finished answers about unchanged; instruction following
unchanged. If no
candidate gains without a per-source loss, nothing is uploaded and the report says so.

## How to run it (for the agent)
- Copy this directory, the bundle's `parent/edit/` and `task.json` to the sandbox's writable space. In the sandbox,
  start the pipeline detached:
  `tmux new -d -s seal "python <this dir>/derive/pipeline.py run --parent <parent/edit/edit.json> --task <task.json> 2>&1 | tee -a /workspace/work/pipeline.log"`
- Check it with `python <this dir>/derive/pipeline.py status` and the logs in `/workspace/work/pipeline/logs/`. It
  runs one job per GPU, keeps `/workspace/work/PROGRESS.md` current, checks every upload with `axonry-check`, uploads
  with `python -m wsp.upload` itself, and with less than 45 minutes left goes straight to the final upload.
- If a step fails, read its log, fix the cause (environment, memory, a path) and run the same command again:
  finished steps are skipped. Do not change the rules (layers, strengths, thresholds, the choice rule); if a step
  cannot run, continue with the rest and record what was skipped in the case's `NOTES.md` (it becomes the report).
- When the final upload is complete, run `axonry-done`. Practice data only: nothing here needs a benchmark, a
  download or the network.
