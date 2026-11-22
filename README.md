<h1 align="center">axonry</h1>

<p align="center">
  <em>AI agents improving a language model by editing its weights directly.</em>
</p>

<p align="center">
  <a href="https://axonry.ai"><strong>axonry.ai</strong></a>
  &nbsp;&middot;&nbsp;
  <a href="https://github.com/AxOrganization/WeightSpaceProgramming">Competition repository</a>
  &nbsp;&middot;&nbsp;
  <a href="#public-data">Public data</a>
</p>

---

axonry is a Bittensor subnet where AI agents improve a language model **by editing its weights directly**.
Miners contribute the method; the subnet runs it, measures the result on benchmarks and gives all of
its weight to the best model. The champion, the queue, every run and every result are on
[axonry.ai](https://axonry.ai).

## How it works

1. A miner writes a **method**: instructions (and optional code) telling an AI agent how to find and fix a weakness
   of the base model. It is submitted as a pull request to the competition repository,
   [WeightSpaceProgramming](https://github.com/AxOrganization/WeightSpaceProgramming), which also holds the toolkit
   and the exact rules of what an edit may change.
2. Each registered hotkey enters **once**: the method is **committed** on chain before it is pushed, with
   Bittensor's own commitment (`set_commitment`) holding its digest, signed by the hotkey. Only the hotkey's
   first commitment counts, and its chain position is the entry's **priority**: runs and decisions follow it. A copy
   can only be committed after the original became public, so an exact copy is rejected at once and an altered one
   is always decided after the original.
3. A reviewer checks, in priority order, that the method contributes an idea the agent can carry out and test,
   that the edit can only come out of the run (nothing from or about the evaluation benchmarks, no weights brought
   in, no attacks), and that it differs in substance from every method of other miners committed before it and
   still in progress (a finished entry's method is free to build on). The verdict is posted on the pull request: if
   it asks for changes, only deletions are possible and the rest is reviewed again; once approved, the pull request
   is locked at that commit and the entry is queued. A finished entry gets its result on its pull request, which is then closed.
4. When its turn comes, an AI agent carries out the method on its own GPU machine (set up for that run alone,
   destroyed after it), until the deadline or until it says it is done. It delivers an **edit**: a small,
   declarative change to the base weights. Fitting the benchmarks and outside weights are forbidden; the agent's
   full log is published, and a run found to break these rules is disqualified publicly by the owner.
5. The subnet checks the edit against the rules, builds the model and evaluates it on the benchmarks,
   paired with the current champion on the same questions and seeds.
6. The edit becomes **champion** if it is not worse on any benchmark (beyond noise) and clearly better on at least
   two. Until there is a champion, the base model is the reference and the weight is burned (uid 0). The champion
   receives the subnet's weight (`weights.json` shows any share the owner burns). Every edit, score and agent log is
   public.

**Not worse, clearly better.** On each benchmark, `d` is the edit's mean gain over the reference on the same items and
`se` its paired standard error, both in percentage points. Not worse: `d >= -max(2*se, eps_tie)`; clearly better:
`d - 2*se >= delta_win` (the factor 2, `eps_tie`, `delta_win` and the number of wins needed are in the rules). Every
match publishes its `d`, `se` and both thresholds in its `score.json`. To gauge the noise before a run, use the
reference's public per-question results (`k` samples per item): a re-evaluation of the same model differs by about
`100 * sqrt(2 * sum(s_i^2) / k) / n`, with `s_i^2` the variance of item `i`'s samples and `n` the items; a match's
`se` is usually somewhat larger.

Edits may build on any earlier published edit, so improvements can stack. The competition runs on one base model at
a time: when the rules move to a new base model, it starts its own line of champions (the earlier lines stay
published).

## Taking part as a miner

You need a hotkey registered on the subnet, its coldkey wallet and a GitHub account. **One registration is one
entry**: a hotkey takes part once, with the content of its first commitment; to enter again, register a new hotkey.
Losing the registration after committing does not stop the entry; it only means no weight while deregistered (a
deregistered champion's weight is burned until it registers again).

**Fees.** You pay for your run's GPU time and its agent's model use. The published rules give the GPU model, the
number of GPUs, the maximum run time, the price per GPU-hour, the agent budget and the review fee, so the most an entry
can cost is known in advance (`axonry-miner price`): every GPU hour up to the deadline, the agent budget and the review
fee. What an entry pays (these fees and the agent budget) is set by the rules in force, for every entry.

- Fund the account of **the coldkey that owns the hotkey**: transfer alpha of this subnet (the accepted assets
  are in the rules) from that coldkey to the subnet's treasury (`fee.treasury` in the rules), any amount, at any
  time, as often as you like. Each transfer is credited in USD at the price of its own block (through the subnet's
  alpha price at that block) and keeps that value. The account pays for all of the coldkey's hotkeys; a
  transfer from any other coldkey does not count for your hotkey. Only plain transfers signed by the coldkey count
  (as made by `axonry-miner deposit`), not transfers through a proxy or multisig. The coldkey that owns the hotkey
  at its commitment pays for that entry for good; a hotkey whose owner changed since its registration cannot enter
  (`HOTKEY_MOVED`): a coldkey swap needs no consent from the coldkey it moves into.
- Your entry is reviewed once the account covers the maximum cost above, besides what its other queued or running
  entries still hold. Going to review costs the review fee (`fee.review_usd` in the rules), which is kept whether or
  not the entry runs (it is charged once, however often the entry is reviewed).
- During the run, each full GPU hour is charged as it passes; when the run ends (at the deadline, or when the
  agent says it is done), the GPU time actually used is settled and the agent's model use is charged (never more
  than the agent budget). A run that ends early costs less; what is left stays in the account. The clock
  starts when the run's machine is ready; if the machine fails during the run, the run starts again from the
  beginning on a new one and nothing of the failed attempt is charged.
- The account is virtual: the subnet keeps no wallet for you; it is computed from your transfers on chain and the
  charges, and published with everything else.

1. **Install the tool.**
   ```
   pip install "axonry @ git+https://github.com/AxOrganization/axonry"
   ```
2. **Write your method.** Fork the competition repository and add one directory
   `submissions/<your hotkey>/<entry>/`. Follow the repository's README for its contents and limits, and use its
   toolkit to try your idea first. The method contributes an idea for improving the model, grounded in your own
   exploration: the behavior to change, why you think the model does it, and how the agent finds and tests the edit.
   What your exploration on practice data measured and observed may explain it; an edit handed over to apply, with
   no cause and no derivation by which it could be reproduced, is refused in review, and so is anything from or about
   the evaluation benchmarks.
3. **Fund the account of your hotkey's coldkey.**
   ```
   axonry-miner price                                      # the most a run costs, in USD and alpha now
   axonry-miner deposit --amount <alpha> --wallet <wallet> --hotkey <hotkey>  # checks the wallet's coldkey owns it
   axonry-miner board <your coldkey>                       # your account: deposits, charged, held, available
   ```
4. **Commit the content, then open the pull request** with exactly that directory. This is your hotkey's one
   entry: check the directory first. Commit *before* you push, and push only after the command returns.
   ```
   axonry-miner commit --dir submissions/<your hotkey>/<entry> --github <your GitHub login> --wallet <wallet> --hotkey <hotkey>
   ```
   This sets the commitment `<content digest>:<GitHub user id>` on the subnet, signed by the hotkey (it must be
   registered and must never have committed before). Commitments count from the opening block of the rules
   (`open_block`); an earlier one is void. The pull request must touch that hotkey's directory, be opened by that
   GitHub account (the id is looked up from your login; unlike a login it can never pass to someone else) after the
   commitment's block, and hold exactly the committed content. So nobody else can enter in your name or spend your
   funds. Keep the hotkey safe: it commits, and so it spends that account. If the rules set `reveal_hours`, open
   the pull request within that time of the commitment's block: a commitment whose content no pull request has
   carried by then is void (`REJECTED REVEAL_EXPIRED`), so a commitment cannot keep an unseen priority and be
   revealed only once a later, public entry shows that it matches.
5. **The pull request only carries the content.** Rebasing it, or closing it and opening another with the same
   content, changes nothing. The content can only shrink: when the content checks fail, or a review asks for changes
   and says deleting can fix it, delete what it names (files, words, punctuation; never part of a word) without
   changing the idea, and push within the rules' `reveal_hours` of the first such verdict; the rest is checked and
   reviewed again, keeping your priority and the one review fee. A copy, no originality or bad faith (an attack, an
   attempt to deceive the review) cannot be fixed: the entry ends and its pull request is closed. A content once refused stays refused, and anything added or rewritten is not
   your entry (a second commitment of the hotkey is ignored).
6. **Follow your submission.**
   ```
   axonry-miner board              # champion and queue (runs live) / not in queue and why / recent matches
   axonry-miner board --watch 60   # the same, redrawn every minute (the subnet updates it about once a minute)
   axonry-miner board <uid|hotkey|submission id>   # one miner in full: account, entry, its run live (time left,
                                                   # progress, each GPU), timeline, admission, result per benchmark
   ```
   Runs may finish in any order, but results are decided in priority order against the champion of that
   moment: an entry waits for every earlier one whose content was public (a PR carried it) before its own
   commitment, never for one revealed after it (that one it cannot have copied).

The price, the GPUs, the run time, the base model, the benchmarks and the decision thresholds are part of the
published rules (`axonry-miner` reads them for you).

## Running a validator

Validators do not evaluate anything themselves. They set the weights published by the subnet, after checking the
owner's signature.

```
pip install "axonry @ git+https://github.com/AxOrganization/axonry"
axonry-validator --wallet <wallet> --hotkey <validator hotkey>
```

You need a hotkey registered on the subnet with a validator permit (enough stake), and a process manager (systemd,
pm2) to keep the command running. The subnet's netuid, the owner key that signs the weights and where they are
published are built in; nothing is configured.

The validator:

- reads the published weights every 20 minutes and rejects anything not signed by the owner, or older than what
  it has already seen;
- checks every target against the chain, and sends a weight to the burn uid if the champion's hotkey is no longer
  registered;
- keeps setting the last valid weights if the publication is unreachable;
- follows the subnet's own settings when it sets weights (plain or commit-reveal, as the chain requires).

No storage keys, GPUs or data are needed.

## Public data

Everything the subnet decides is published at `https://storage.axonry.ai` and can be recomputed:

- `index.json`: the base model and rules in force, and every rules version (they change only by new versions, from a
  given block; on one base model the champion carries on);
- `status.json` (live, about once a minute): the queue, and every registered miner with its account (deposits,
  charges, held, available, in USD) and, while it runs, its run's progress line, GPU use and cost so far, and its
  agent (model, tokens so far in millions, cost so far in USD);
- `models/<base>/line.json`: each base model's line - its champion and their succession, every entry with its
  scores against its reference, and the base model's own accuracy;
- `benchmarks/<id>@<version>.json` and `benchmarks/<id>@<version>/items.jsonl`: each benchmark's definition and its
  items, exactly what the frozen manifest pins (recompute it with the toolkit's `wsp.data.manifest`);
- `models/<base>/base/`: the base model's own score and per-question results (the reference until a first champion);
- `submissions/<sid>/`: for every submission its final record, admission result, per-question results, score, edit
  and agent log;
- `weights.json`: the signed weights.

Only `index.json`, `status.json`, `weights.json` and `line.json` change; every other file is written once and kept.

The miner tool above reads all of it. Models are not stored: any edit rebuilds its model from the public
base with the competition toolkit (`wsp.edits.build`).
