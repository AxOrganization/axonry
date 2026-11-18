# Review protocol

How a submission goes from pull request to run, and who does what. Three independent parties, each with one job:

| Party | Reads | Writes | Holds |
|---|---|---|---|
| Miner | - | on chain: commitments (hotkey), deposits (coldkey); GitHub: the PR | coldkey/hotkey, GitHub account |
| Reviewer (`axonry-review run`) | `status.json`, GitHub | GitHub reviews only | a review account token |
| Orchestrator | chain, GitHub, its DB | DB, lock tags, merges, public files | all operating keys |

The reviewer never sees the orchestrator's state; the orchestrator trusts a review only from an account listed in the
rules' `review.reviewers`, and only for the content it was made on.

## 1. Before review (orchestrator)

A miner is a hotkey, and one registration is one entry. The orchestrator reads every finalized block; a hotkey's
first commitment on the subnet (`set_commitment` `<digest>:<GitHub id>`, which the chain only takes from a
registered hotkey) is its submission, for good: `WAITING`, with its sid and priority = that chain position. Any
later commitment of the hotkey counts for nothing. Commitments before the rules' `open_block` are void. A digest committed earlier by another hotkey is a copy:
`REJECTED COPY` at once.

A waiting submission goes to review (`REVIEW`) when
- an open PR carries it: in `submissions/<hotkey>/<entry>/`, by the committed GitHub id, opened after the
  commitment's block, with exactly the committed content (the PR is only the carrier: it may be rebased, or closed
  and another opened),
- if the rules set `reveal_hours`, the first PR carrying it was opened (GitHub's time) within that many hours
  of the commitment's block; once that time has passed with no such PR, the entry is void: `REJECTED
  REVEAL_EXPIRED` (final). Without it a commitment could keep its priority unseen and be revealed only once a
  later, public entry shows that it matches, taking review and decision ahead of that entry,
- the content passes the static checks and names a published parent if any (invalid content can never be fixed:
  `REJECTED`),
- and the account of the coldkey that owned the hotkey at its commitment (fixed then; a hotkey whose owner changed
  since its registration is `REJECTED HOTKEY_MOVED` at once; deposits come from that coldkey, each in USD at the
  price of its block) covers a whole run at its maximum cost, besides what the coldkey's other locked entries hold.
  Going to review then costs the review fee (`fee.review_usd`): part of its run if it runs, kept if it never runs.

So nobody can enter in someone else's name (only the hotkey signs, and the commitment names the GitHub account),
spend someone else's funds (only the owner's account pays), enter an exact copy, or hold an unseen priority
beyond the reveal deadline.

## 2. Review (reviewer)

The reviewer takes `status.json` `review` top-down (priority order). For the first item that has no verdict on its
content yet:

1. Fetch the directory at the listed `commit`; its digest must equal the listed `content_digest` (else skip: the
   status is stale).
2. Build a workspace: `submission/` and `earlier/` = every entry of `status.json` `contents` with an earlier
   priority (every submission still in progress whose content a PR has made public; a finished entry's PR is
   closed, and its method is free to build on).
3. Run the model on `POLICY.md` in a read-only sandbox without network and without any token in its environment.
   It answers three checks - **substantive**, **conduct**, **original** - in `verdict.schema.json`.
4. Post one GitHub review bound to that commit: `APPROVE` when all three pass, otherwise `REQUEST_CHANGES` with
   the reasons. That review is the public reply to the miner.

An item whose review fails three times (model error, malformed verdict) is skipped and left to a human, who uses
`axonry-review show/approve/changes`. Review follows priority but does not wait for an earlier item that is not
reviewable (no PR, no funds): copies are caught by comparing with all earlier content still in progress, and decisions are
ordered by priority regardless (sections 3-4).

## 3. Lock and queue (orchestrator)

On every tick, for each submission in `WAITING` or `REVIEW`, the orchestrator reads its carrier PR:

| Review on the committed content | Result |
|---|---|
| none yet | `REVIEW` |
| changes requested | `REJECTED REVIEW_CHANGES` (the reviewer's text); the review fee is kept; the hotkey's entry is spent |
| approved, PR conflicts with the main branch | `WAITING CONFLICT` (rebase: the content, so the approval, stays) |
| approved | **locked** (tag `axonry/<sid>` at that commit), holding the run's maximum cost, `QUEUED` |

From the lock on, pushes to the PR never affect the entry.

## 4. Runs and decisions (orchestrator)

- `QUEUED` submissions get a GPU machine each in priority order (up to the concurrency limit); while it is set up
  status.json shows `STARTING`, and the run starts (`RUNNING`) when it is ready; several run at once and may finish
  in any order.
- A finished entry (rejected, failed or decided) gets its result as a comment on its PR, which is then closed.
- A run is charged for its GPUs by the hour (one hour's price at each full hour from its start, and at its end -
  the deadline, or the agent's `axonry-done` - the time actually used) and, at its end, for its agent's model use
  (at most the agent budget); never more than its account was holding.
- Decisions do not follow finishing order. Entries are walked in priority order: a scored submission is judged
  against the champion of that moment, and the walk stops at the first earlier submission still in review, queued,
  running or being evaluated. A later commitment - a copy that slipped through, or a lucky run - is never judged
  before the earlier content it may have copied. `WAITING` submissions (no PR or no funds yet) do not hold the
  walk: a miner who does not show up cannot stall everyone.
- A new champion's locked commit is merged into the main branch; every PR gets its result as a comment.
- Each base model has its own line of champions (rules moving to a new base start a new line; earlier lines stay
  published). Until a line has a champion, its base model is the reference and the weight goes to uid 0 (burned).
- A run found afterwards to break the rules (e.g. its public agent log shows benchmark data or outside weights) is
  disqualified by the owner, publicly (`ruling.json`); a champion's title goes back to the previous one.
