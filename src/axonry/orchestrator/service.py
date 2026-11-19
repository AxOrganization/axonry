"""The orchestrator: one process per role, a loop of idempotent ticks over the DB state machine.

One registration, one entry: a hotkey's first commitment on the subnet is its submission; its chain position
(block, extrinsic) is the sid and the priority, fixed for good. Runs and decisions follow it.

  intake    finalized blocks -> deposits (credited in USD at their own moment's price to the sender coldkey's
            account) and participants (a hotkey's first commitment -> its submission, WAITING; an exact copy of
            earlier content -> REJECTED COPY)
  advance   WAITING/REVIEW: a PR carrying the committed content? valid content? the account covers a whole run?
            -> REVIEW (the review fee is charged); review: changes -> REJECTED (the fee is kept); approved -> locked (tag axonry/<sid>) -> QUEUED
  schedule  QUEUED in priority order: a rented machine each (up to max_concurrent); when it is ready, the run's
            agent gets it + R2 creds -> RUNNING. A run whose machine vanishes runs again, nothing charged
  supervise RUNNING -> agent steps until the deadline (or an early finish), charged each full hour; export log;
            machine released; the used time settled -> COLLECTING
  eval_machine  the evaluation machines (eval_machines at most): one while there is admission or evaluation
            work, all while the evaluation queue is long; released when idle or no longer needed
  collect   staging upload -> admission on an evaluation machine                -> ADMITTED | REJECTED | FAILED
  evaluate  ADMITTED -> benchmarks, one model per evaluation machine at a time (the base model first, once: the
            reference)                                                           -> SCORED
  decide    SCORED in priority order vs the champion of the moment               -> PUBLISHED (+ weights, merge)
  janitor   staging uploads of finished submissions; old state snapshots
The competition runs on one base model at a time; each base has its own line of champions (registry.py).
Every external system is injected (see Deps), so the whole flow runs against fakes in tests.

Status values (public): WAITING REVIEW QUEUED RUNNING COLLECTING ADMITTED SCORED PUBLISHED REJECTED FAILED
DISQUALIFIED
Reason codes (public, stable): intake.binding (NO_PR, INSUFFICIENT_FUNDS, COPY, REVEAL_EXPIRED, ...); CONFLICT,
REVIEW_CHANGES, BAD_PARENT; NO_UPLOAD, BAD_UPLOAD, REJECTED_RULES, EVAL_ERROR; decisions NEW_CHAMPION, NOT_DOMINATING.
"""
import io
import json
import logging
import shutil
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import judge, pricing, registry
from ..agent import brief, supervisor
from ..agent.base import RunHandle
from ..chain.extrinsics import Deposit
from ..common import layout
from ..common.canon import pretty
from ..common.store import ImmutableObjectError
from ..common.ids import make_sid
from ..evaluation import staging
from ..evaluation.executor import JobFailed
from ..review.auto import fixable as review_fixable
from ..intake import binding
from ..pool import NoMachine
from ..pool.machine import eval_names, is_eval, keypair
from ..publish import publisher, redact
from .db import DB

log = logging.getLogger("axonry")
REVIEW_FEE, REVIEW_FEE_BACK = -2, -3      # charge numbers (db.charge)


def _utc(ts) -> str:
    """A public reason's time: unix seconds -> 'YYYY-MM-DD HH:MM UTC'."""
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ts))


def without_secrets(task: dict) -> dict:
    """The task as kept at rest (DB and its snapshots): the sandbox key and the R2 credentials are only ever in the
    bundle handed to the agent."""
    t = json.loads(json.dumps(task))
    t["sandbox"].pop("private_key", None)
    for k in ("secret_access_key", "session_token"):
        t["r2"].pop(k, None)
    return t


@dataclass
class Deps:
    db: DB
    chain: object                 # .finalized_head() .items_in_block(b, treasury, netuid) .owner_of .uid_of .registration_owner .alpha_price .uid_hotkeys
    github: object                # intake.github.GitHub-like: scan(), dir_blobs(commit, dir)
    fleet: object                 # pool.Machines: owned hosts, then vast.ai (runs, evaluation)
    issue_creds: object           # (store, prefix, ttl) -> creds.TempCreds for that prefix of that store
    runner: object                # agent.base.AgentRunner
    executor: object              # evaluation.executor.SshExecutor-like
    public: object                # ObjectStore (public bucket)
    private: object               # ObjectStore (private bucket)
    owner_keypair: object
    comp_rules: dict              # competition repo rules.json at the pinned commit
    price_at: object              # ts -> (tao_usd, sources)
    scan_prs: object              # () -> [PRCandidate] (open PRs, head content)
    clock: object = time.time
    registry_root: Path = registry.ROOT
    work_dir: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "axonry-work")
    max_concurrent: int = 4       # runs holding a machine at once, from any source (also bounded by the agent backend)
    eval_gpu_name: str = "RTX 4090"
    eval_gpus: int = 8
    eval_idle_minutes: int = 20
    eval_machines: int = 1        # evaluation machines at most, from any source
    eval_scale_up: int = 10       # all of them while more models than this wait for evaluation ...
    eval_scale_down: int = 5      # ... until fewer than this do; otherwise one
    burn: float = 0.0             # the owner's share of the weight burned (uid0) beside the champion's
    eval_image: str | None = None # the pinned container image results are generated in (recorded with each score)


class Orchestrator:
    def __init__(self, d: Deps):
        self.d, self.db = d, d.db

    # ------------------------------------------------------------------------------------------- helpers
    def rules_for(self, sid) -> registry.Rules:
        """The rules version a submission entered under (recorded at entry; a later version never applies to it)."""
        rec = self.db.sub(sid)["record"]
        for r in registry.load_rules_versions(self.db.sub(sid)["base"], self.d.registry_root):
            if r.sha256 == rec["rules_sha256"]:
                return r
        raise registry.RegistryError(f"{sid}: rules {rec['rules_sha256'][:12]} not in the registry")

    def now(self):
        return int(self.d.clock())

    def _reject(self, sid, code, detail, final_status="REJECTED"):
        charged = self.db.charged(sid)        # e.g. the review fee of an entry refused in review: kept in its record
        patch = {"fee": {**self.db.sub(sid)["record"].get("fee", {}), "charged_usd": sum(charged.values()) / 1e6}}
        self.db.update(sid, final_status, reason_code=code, reason=detail, decided_at=self.d.clock(),
                       record_patch=patch if charged else None)
        self._publish_record(sid)
        self._publish_line(self.db.sub(sid)["base"])
        self._close_pr(sid, f"axonry: submission `{sid}` -> **{final_status} {code}**: {detail[:1000]}")

    FINAL = ("PUBLISHED", "REJECTED", "FAILED", "DISQUALIFIED")

    def _close_pr(self, sid, text, merge_commit=None):
        """A finished entry's PR: the result as a comment, a champion's locked commit merged, and the PR closed (an
        open PR is an entry still in progress: later entries are compared only with those). Recorded first and
        carried out step by step until done (settle_prs: GitHub may fail any of them)."""
        if self.db.sub(sid)["pr_number"]:
            self.db.update(sid, record_patch={"pr_result": {"text": text, "merge": merge_commit}})
            self._settle_pr(self.db.sub(sid))

    def settle_prs(self):
        for s in self.db.subs(status=self.FINAL):
            r = s["record"].get("pr_result")
            if r and not r.get("closed"):
                self._settle_pr(s)

    def _settle_pr(self, s):
        r, n = dict(s["record"]["pr_result"]), s["pr_number"]
        try:
            if not r.get("commented"):
                self.d.github.comment(n, r["text"])
                r["commented"] = True
            if r.get("merge") and not r.get("merged"):   # the locked commit itself, whatever the PR head is now
                self.d.github.merge(r["merge"], f"Champion {s['sid']} (PR #{n})")
                r["merged"] = True
            self.d.github.close(n)
            r["closed"] = True
        except Exception:
            log.exception("PR #%s of %s (tried again next tick)", n, s["sid"])
        self.db.update(s["sid"], record_patch={"pr_result": r})

    # ------------------------------------------------------------------------------------------- intake
    def intake(self):
        """Read finalized blocks in order. A deposit is credited to its sender coldkey's account, in USD at the price
        of its own block (until a price is known it is pending and counts for nothing). A commitment is a
        participant: a hotkey's first commitment on the subnet (the chain only takes one from a registered hotkey)
        becomes its one and only submission; any later commitment of that hotkey counts for nothing. Then every
        submission before its lock is advanced."""
        head = self.d.chain.finalized_head()
        cursor = self.db.get("cursor")
        if cursor is None:       # deposits and commitments before the first rules took effect are never read
            first = registry.all_rules(self.d.registry_root)[0]
            if first["effective_block"] <= 0:
                raise ValueError("the first rules have no effective_block yet (the block they take effect at)")
            cursor = first["effective_block"] - 1
            self.db.set("cursor", cursor)
        for b in range(cursor + 1, head + 1):
            try:
                r = registry.rules_at(b, self.d.registry_root)
            except registry.RegistryError:
                self.db.set("cursor", b)
                continue
            for x in self.d.chain.items_in_block(b, r["fee"]["treasury"], r["netuid"]):
                if isinstance(x, Deposit):
                    if x.asset in r["fee"]["assets"]:
                        self.db.add_deposit(x)
                else:
                    self._participant(x, r)
            self.db.set("cursor", b)
        self.price_deposits()
        self.advance_all(head)

    def price_deposits(self):
        """Each deposit once, at its own moment: TAO/USD at its block time; alpha via the alpha price at its block."""
        for d in self.db.unpriced_deposits():
            try:
                tao_usd, sources = self.d.price_at(d["ts"])
                alpha_tao = None
                if d["asset"] == "alpha":
                    alpha_tao = self.d.chain.alpha_price(registry.rules_at(d["block"], self.d.registry_root)["netuid"],
                                                         d["block"])
                usd = pricing.deposit_musd(d["amount"], d["asset"], tao_usd, alpha_tao)
            except (pricing.PriceError, registry.RegistryError) as e:
                log.warning("deposit %s-%s: no price yet (%s)", d["block"], d["idx"], e)
                continue
            self.db.price_deposit(d["block"], d["idx"], d["asset"], usd,
                                  {"tao_usd": tao_usd, "tao_usd_sources": sources, "alpha_tao": alpha_tao})

    def _participant(self, c, rules):
        """One registration, one entry: the hotkey's first commitment is its submission (sid = its chain position,
        which is also its priority, fixed for good). A content already committed by another hotkey is a copy.
        Commitments before the rules' open_block count for nothing. The owner's hotkey (its weight is burned) is
        bound by neither.
        The entry is paid by the coldkey that owned the hotkey at the commitment, for good: nobody can move a hotkey
        under someone else's coldkey (a coldkey swap needs no consent from its destination) and spend that account.
        For the same reason a hotkey whose owner changed since its registration cannot enter. Every chain read
        happens before the entry is recorded: a failed read is retried with the block, never skipped."""
        owner = c.signer == self.d.owner_keypair.ss58_address
        sid = make_sid(c.block, c.extrinsic_index)
        if self.db.sub(sid) or (not owner and (self.db.of_hotkey(c.signer) or c.block < rules.doc.get("open_block", 0))):
            return                                  # already read (a block is read again after a failed read)
        rec = {"hotkey": c.signer, "rules_sha256": rules.sha256,
               "content_digest": c.content_digest, "github_id": c.github_id, "timeline": {"committed": c.timestamp}}
        coldkey = self.d.chain.owner_of(c.signer, c.block)
        reg, reg_owner = (None, coldkey) if owner or coldkey is None else \
            self.d.chain.registration_owner(c.signer, rules["netuid"], c.block)
        first = self.db.first_with_digest(c.content_digest)
        self.db.add_submission(sid, rules["base"], c.signer, coldkey, "WAITING", rec, content_digest=c.content_digest,
                               prio=(c.block, c.extrinsic_index))
        if first:
            self._reject(sid, binding.COPY, f"this exact content was committed first by {first['hotkey']} ({first['sid']})")
        elif coldkey is None:
            self._reject(sid, binding.HOTKEY_UNKNOWN, f"{c.signer} had no owning coldkey at its commitment")
        elif reg is not None and reg_owner != coldkey:
            self._reject(sid, binding.HOTKEY_MOVED, f"the hotkey's owner changed since its registration (block {reg}): "
                         "its entry cannot be paid by its new owner")

    # ------------------------------------------------------------------------------------------- advance
    PRE_LOCK = ("WAITING", "REVIEW")

    def advance_all(self, head):
        """Every submission before its lock is moved as far as it can go (advance); PRs that carry nothing are listed
        with the reason (status.json "prs")."""
        prs, used = self.d.scan_prs(), set()
        for s in sorted(self.db.subs(status=self.PRE_LOCK), key=self.priority):
            try:
                n = self.advance(s, prs, head)
                if n:
                    used.add(n)
            except Exception:
                log.exception("advance %s", s["sid"])
        used |= {s["pr_number"] for s in self.db.subs() if s["pr_number"]}    # a PR shown with its submission
        self.db.set("prs", {str(p.number): self._idle_pr(p) for p in prs if p.number not in used})

    def _idle_pr(self, p):
        s = self.db.of_hotkey(p.hotkey) if p.hotkey else None
        if p.problem:
            code, why = binding.BAD_PR, p.problem
        elif s is None:
            code, why = binding.UNCOMMITTED, "this hotkey has no commitment on the subnet (commit, then push)"
        elif s["status"] not in self.PRE_LOCK:
            code, why = binding.PARTICIPATED, f"this hotkey's one entry is {s['sid']} ({s['status'].lower()})"
        else:
            code, why = binding.UNCOMMITTED, binding.carrier_problem(s, p, self._content(s))
        return {"pr": p.number, "hotkey": p.hotkey, "title": p.title, "reason_code": code, "reason": why}

    def _await_fix(self, s, code, detail, hours):
        """WAITING for deletions that fix the content: within `hours` of the first problem, when set (never extended)."""
        by = s["record"].get("fix_by")
        if by is None:
            by = int(self.now() + float(hours) * 3600) if hours is not None else 0
            self.db.update(s["sid"], record_patch={"fix_by": by})
            s = self.db.sub(s["sid"])
        if by and self.now() > by:
            return self._reject(s["sid"], code, f"nothing fixed it by {_utc(by)}: {detail}"[:2000])
        self._set_stage(s, "WAITING", code, f"{detail}. Only deletions are possible: delete what this names, keeping "
                        "the idea, and push" + (f" by {_utc(by)}, or the entry ends" if by else ""))

    @staticmethod
    def _content(s) -> str:
        """The digest of the content the submission carries now: the committed one, or a reduction of it."""
        return s["record"].get("content") or s["content_digest"]

    @staticmethod
    def _revealed(s) -> dict | None:
        """Where the committed content was first carried: {commit, directory} (what a reduction is checked against)."""
        return s["record"].get("revealed") or (     # entries carried before reductions existed: their last head
            s["head_sha"] and {"commit": s["head_sha"], "directory": s["record"]["pr"]["directory"]}) or None

    def _reduces(self, s, p) -> bool:
        """p's content is the entry's current content (at its last head) with files, words or punctuation deleted."""
        if not s["head_sha"] or p.digest is None:
            return False
        gh = self.d.github
        return binding.reduces(gh.dir_blobs(p.head, p.directory),
                               gh.dir_blobs(s["head_sha"], s["record"]["pr"]["directory"]))

    def advance(self, s, prs, head):
        """WAITING / REVIEW -> what the submission still lacks, in this order; -> the number of its PR, if any.
          a PR carrying it   an open PR in the hotkey's directory, by the committed GitHub id, with the committed
                             content (the PR is only the carrier: it can be rebased, or closed and another opened)
          revealed in time   with the rules' reveal_hours, the first PR carrying it is opened within that time of
                             the commitment; past it with no such PR the entry is void (REJECTED REVEAL_EXPIRED)
          valid content      entry.json and files; invalid -> WAITING with the problem, to be fixed by deletions
          funds              the account of the coldkey owning the hotkey covers a whole run (besides what its
                             other locked entries hold); then the review fee is charged, once (it counts towards
                             the run, or is kept if review refuses the entry)
          the review         of the content the PR carries: none -> REVIEW; changes requested -> WAITING
                             REVIEW_CHANGES (a content refused once stays refused, on any PR)
          fixing             after invalid content or changes requested, and only then, a PR with the revealed
                             content minus files, words or punctuation carries the entry (keeping its priority) and is checked and
                             reviewed again; with the rules' reveal_hours, all fixing has that long from the first
                             problem (never extended): content still refused or invalid after it ends the entry
                             (meanwhile its originality stands: later entries are compared with it)
          approved           -> locked (tag axonry/<sid>), holding the run's maximum cost, QUEUED"""
        sid, rec = s["sid"], s["record"]
        rules = self.rules_for(sid)
        fixing = rec.get("fix_by") is not None
        pr = next((p for p in sorted(prs, key=lambda p: -p.number)          # the newest carrier
                   if binding.carrier_problem(s, p, self._content(s),
                                              lambda p: fixing and self._reduces(s, p)) is None), None)
        fix_hours = None if s["hotkey"] == self.d.owner_keypair.ss58_address else rules.doc.get("reveal_hours")
        unfixed = self._content(s) in (rec.get("refused") or {}) or self._content(s) == rec.get("invalid")
        if fixing and rec["fix_by"] and self.now() > rec["fix_by"] and (pr is None or (
                pr.digest == self._content(s) and unfixed)):
            self._reject(sid, s["reason_code"], f"nothing fixed it by {_utc(rec['fix_by'])}: {s['reason'] or ''}"[:2000])
            return pr.number if pr else None
        # reveal deadline: until a PR carries it nobody sees the content its priority belongs to; without a deadline a
        # miner could commit several contents and reveal only the one a later, public entry shows to match (once a PR
        # has carried it the content is public; the owner's hotkey is not bound)
        hours = None if s["pr_number"] or s["hotkey"] == self.d.owner_keypair.ss58_address \
            else rules.doc.get("reveal_hours")
        by = None if hours is None else int(rec["timeline"]["committed"] + float(hours) * 3600)
        if pr is None:
            if by is not None and self.now() > by:
                self._reject(sid, binding.REVEAL_EXPIRED, f"no PR carried the committed content by {_utc(by)} "
                             f"({float(hours):g} h after the commitment)")
                return None
            return self._set_stage(s, "WAITING", binding.NO_PR, "no open PR carries the committed content yet "
                                   "(see status.json prs for PRs in this hotkey's directory)"
                                   + (f"; open it by {_utc(by)} or the entry is void" if by is not None else ""))
        if s["pr_number"] != pr.number or s["head_sha"] != pr.head:
            patch = {"pr": {"repo": getattr(self.d.github, "repo", None), "number": pr.number, "author": pr.author,
                            "directory": pr.directory, "title": pr.title, "head": pr.head}, "content": pr.digest}
            if not rec.get("revealed"):         # what may only shrink: the committed content as first carried
                first = self._revealed(s) if s["head_sha"] else (       # (carried before reductions existed)
                    {"commit": pr.head, "directory": pr.directory} if pr.digest == s["content_digest"] else None)
                if first:
                    patch["revealed"] = first
            self.db.update(sid, pr_number=pr.number, head_sha=pr.head, record_patch=patch)
            s = self.db.sub(sid)
        if by is not None and pr.opened_at is not None and pr.opened_at > by:     # GitHub's time, not our tick's
            self._reject(sid, binding.REVEAL_EXPIRED, f"PR #{pr.number} was opened at {_utc(pr.opened_at)}, after "
                         f"the reveal deadline {_utc(by)} ({float(hours):g} h after the commitment)")
            return pr.number
        entry = binding.check_content(self.d.github.dir_blobs(pr.head, pr.directory), rules, self.d.comp_rules)
        if isinstance(entry, binding.Rejected):
            if rec.get("invalid") != self._content(s):
                self.db.update(sid, record_patch={"invalid": self._content(s)})
                s = self.db.sub(sid)
            self._await_fix(s, entry.code, entry.detail, fix_hours)
            return pr.number
        why, final = self._parent_problem(entry)
        if why:
            if final:
                self._reject(sid, "BAD_PARENT", why)
            else:
                self._set_stage(s, "WAITING", "BAD_PARENT", why)
            return pr.number
        coldkey = s["coldkey"]                  # fixed at the commitment
        short = self._funds(s, rules)
        if short:
            self._set_stage(s, "WAITING", *short)
            return pr.number
        fee = pricing.review_musd(rules)
        if fee and s["hotkey"] != self.d.owner_keypair.ss58_address:   # going to review costs the review fee, once
            self.db.charge(sid, REVIEW_FEE, coldkey, fee)
        refused = s["record"].get("refused") or {}
        verdict, body = (("changes", refused[self._content(s)]) if self._content(s) in refused else   # on any PR
                         self.d.github.review_state(pr.number, self._content(s), pr.directory,
                                                    rules["review"]["reviewers"]))
        if verdict == "changes":
            summary = (body or "changes requested").strip().split("\n\n")[0]   # the verdict; details stay on the PR
            if not review_fixable(body):          # no deletion can fix it (e.g. a copy): the entry ends, its PR closes
                self._reject(sid, "REVIEW_CHANGES", f"{summary[:600]} (full review on PR #{pr.number}; not fixable by "
                             "deletion)")
                return pr.number
            if self._content(s) not in refused:
                self.db.update(sid, record_patch={"refused": {**refused, self._content(s): summary[:600]}})
                s = self.db.sub(sid)
            self._await_fix(s, "REVIEW_CHANGES", f"{summary[:600]} (full review on PR #{pr.number})", fix_hours)
        elif verdict is None:
            self._set_stage(s, "REVIEW", None, None)
        elif self.d.github.pr(pr.number)["mergeable"] is False:
            self._set_stage(s, "WAITING", "CONFLICT", "the PR conflicts with the main branch (rebase it)")
        else:
            try:
                self.d.github.lock_tag(sid, pr.head)
            except Exception as e:      # GitHub unreachable: ours, the approval stands, retried next tick
                log.warning("lock %s: %s (retrying)", sid, e)
                return pr.number
            self.db.lock(sid, pr.head, coldkey, pricing.run_cost_musd(rules), record_patch={
                "pr": {**self.db.sub(sid)["record"]["pr"], "locked_commit": pr.head, "lock_tag": f"axonry/{sid}"},
                "coldkey": coldkey, "base": entry["base"], "parent": entry.get("parent"),
                "fee": {"usd_per_gpu_hour": rules["fee"]["usd_per_gpu_hour"], "gpus": rules["run"]["gpus"],
                        "max_hours": rules["run"]["hours"], "max_agent_usd": rules["run"]["max_agent_usd"],
                        "max_usd": pricing.run_cost_musd(rules) / 1e6},
                "timeline": {**rec.get("timeline", {}), "entered": self.now()}})
            self._set_stage(self.db.sub(sid), "QUEUED", None, None)
        return pr.number

    def _funds(self, s, rules):
        """-> None when the account of the coldkey paying for the entry covers a whole run, else (code, detail)."""
        paid = sum(self.db.charged(s["sid"]).values())          # its review fee, once charged, is part of the run
        if self.db.available(s["coldkey"], besides=s["sid"]) < pricing.run_cost_musd(rules) - paid:
            return binding.INSUFFICIENT_FUNDS, "insufficient funds"
        return None

    def _set_stage(self, s, status, code, reason):
        if (s["status"], s["reason_code"], s["reason"]) != (status, code, reason):
            self.db.update(s["sid"], status, reason_code=code, reason=reason)
            self._publish_line(s["base"])

    @staticmethod
    def priority(s):
        return (s["prio_block"], s["prio_index"], s["sid"])

    def _parent_problem(self, entry):
        """A stacking parent must be a published (not disqualified) submission on the same base.
        -> (problem or None, final: the parent can never qualify)."""
        parent = entry.get("parent")
        if not parent:
            return None, False
        p = self.db.sub(parent)
        if p and p["status"] == "PUBLISHED" and p["record"].get("base") == entry["base"]:
            return None, False
        final = p is None or p["status"] in ("REJECTED", "FAILED", "DISQUALIFIED") or (
            p["status"] == "PUBLISHED" and p["record"].get("base") != entry["base"])
        return f"parent {parent} is not a published submission on base {entry['base']}", final

    # ------------------------------------------------------------------------------------------- schedule
    def schedule(self):
        """QUEUED (locked) submissions get a machine each in priority order, while fewer than max_concurrent runs hold
        one (an entry that waits for a machine keeps every later one waiting too, but never stops a machine already
        rented from being set up); a run starts when its machine is ready (setup time is ours). Runs may finish out of order; decisions
        never do (see decide). A machine whose run is over is released; a run whose machine vanished runs again."""
        live = {s["sid"]: s for s in self.db.subs(status=("QUEUED", "RUNNING"))}
        for m in self.d.fleet.active():
            s = live.get(m["name"])
            if not is_eval(m["name"]) and s is None:
                self.d.fleet.release(m["name"])
            elif s and s["status"] == "RUNNING" and self.d.fleet.vanished(m["name"]):
                self._run_lost(s, "its machine vanished")
        held = sum(1 for m in self.d.fleet.active() if not is_eval(m["name"]))
        renting = True          # machines are rented in priority order: once one cannot be, none after it is
        for s in sorted(self.db.subs(status=("QUEUED",)), key=self.priority):
            if not self._due(s):
                continue
            sid, rules = s["sid"], self.rules_for(s["sid"])
            if self.d.fleet.get(sid) is None:
                if not renting or held >= self.d.max_concurrent:
                    renting = False
                    continue
                try:
                    self.d.fleet.rent(sid, rules["run"]["gpu_model"], rules["run"]["gpus"])
                    held += 1
                except NoMachine:       # all busy: the queue waits, in order (machines already held go on)
                    renting = False
                    continue
                except Exception as e:  # no offer, vast.ai down, a host unreachable: ours, retried
                    self._backoff(s, f"renting a machine: {e}")
                    continue
            state = self.d.fleet.advance(sid)
            if state == "failed":
                self._backoff(s, "machine setup failed")
            if state != "ready":
                continue
            try:
                self._provision(s, rules)
                self._recovered(s)
            except Exception as e:  # ours: retried on the same machine (starting a run is repeatable); a machine
                log.exception("provision %s failed", sid)       # that fails it three times in a row is replaced
                self._backoff(s, f"provision: {e}")
                if self.db.get(f"backoff:{sid}:n", 0) >= 3:
                    self.d.fleet.release(sid, f"provision failed: {e}"[:300])

    def _run_lost(self, s, why):
        """Our infrastructure lost the run (its machine vanished, the agent backend could not run it): the agent is
        stopped, the charges are refunded, the staging area cleared, and the submission is queued again with its
        priority (it runs again from the start, on a new machine)."""
        sid = s["sid"]
        log.warning("%s: %s; the run starts again", sid, why)
        try:
            self.d.runner.stop(RunHandle(**self.db.run(sid)["handle"]))
        except Exception:
            log.exception("stop agent of %s", sid)
        self.d.fleet.release(sid, why[:200])
        self.db.refund(sid)
        self.d.private.delete_prefix(layout.staging_prefix(sid))
        tl = {k: v for k, v in s["record"]["timeline"].items() if k not in ("kickoff", "deadline")}
        self.db.update(sid, "QUEUED", record_patch={"timeline": tl})

    def _provision(self, s, rules):
        sid, rec = s["sid"], s["record"]
        blobs = self.d.github.dir_blobs(s["locked_commit"], rec["pr"]["directory"])
        run = rules["run"]
        start = self.now()
        deadline = start + int(run["hours"] * 3600)
        ttl = deadline - start
        private_key, public_key = keypair(f"axonry-{sid}")
        m = self.d.fleet.hand_over(sid, public_key, deadline)
        resources = self.d.fleet.probe(sid)
        creds = self.d.issue_creds(self.d.private, layout.staging_prefix(sid), ttl)
        task = {"schema": "axonry.task/1", "sid": sid, "hotkey": s["hotkey"],
                "base": rec["base"], "rules_sha256": rules.sha256,
                "upload_protocol": self.d.comp_rules["upload"]["schema"],
                "run": {"start": start, "deadline": deadline, "wrap_minutes": run["wrap_minutes"],
                        "stall_minutes": run["stall_minutes"], "max_agent_usd": run.get("max_agent_usd")},
                "sandbox": {"host": m.ip, "port": m.port, "user": m.user, "host_key": m.host_key(),
                            "private_key": private_key, "resources": resources,
                            "jump": m.jump and {"host": m.jump.host, "port": m.jump.port, "user": m.jump.user,
                                                "host_key": m.jump.host_key()}},
                "r2": creds.as_dict(),
                "limits": brief.limits_from(self.d.comp_rules),
                "parent": ({"sid": rec["parent"], "path": "parent/edit/edit.json"} if rec.get("parent") else None)}
        bundle = self._bundle(task, blobs, rec.get("parent"))   # the only place the run's secrets go
        busy = [RunHandle(**self.db.run(r["sid"])["handle"]) for r in self.db.subs(status=("RUNNING",))]
        h = self.d.runner.prepare(sid, s["hotkey"], busy)
        self.d.runner.kickoff(h, brief.kickoff_message(bundle.name), bundle)
        self.db.save_run(sid, task=without_secrets(task), handle=asdict(h), sup_state={})
        self.db.update(sid, "RUNNING", record_patch={
            "timeline": {**rec["timeline"], "kickoff": start, "deadline": deadline},
            "sandbox": {"resources": resources}})

    def _parent_files(self, parent_sid) -> dict:
        """{relative path: bytes} of a published submission's edit/ directory (the stacking parent)."""
        prefix = layout.submission_edit_prefix(parent_sid)
        return {k[len(prefix):]: self.d.public.get(k) for k, _ in self.d.public.list(prefix)}

    def _bundle_path(self, sid) -> Path:
        return Path(self.d.work_dir) / sid / f"axonry-task-{sid}.tar.gz"

    def _bundle(self, task, blobs, parent_sid=None) -> Path:
        d = Path(self.d.work_dir) / task["sid"]
        d.mkdir(parents=True, exist_ok=True)
        path = self._bundle_path(task["sid"])
        with tarfile.open(path, "w:gz") as tf:
            def add(name, data):
                ti = tarfile.TarInfo(name)
                ti.size, ti.mtime, ti.mode = len(data), 0, 0o600
                tf.addfile(ti, io.BytesIO(data))
            add("task.json", brief.task_json(task))
            add("BRIEF.md", brief.render(task).encode())
            for rel, data in brief.ssh_files(task).items():
                add(f"ssh/{rel}", data)
            for rel, data in sorted(blobs.items()):
                add(f"submission/{rel}", data)
            if parent_sid:     # the full published chain, nested parent/ directories included
                chain = d / "parent_chain"
                shutil.rmtree(chain, ignore_errors=True)
                self._materialize_chain(parent_sid, chain)
                for p in sorted(chain.rglob("*")):
                    if p.is_file():
                        add(f"parent/edit/{p.relative_to(chain).as_posix()}", p.read_bytes())
        return path

    # ------------------------------------------------------------------------------------------- supervise
    def supervise(self):
        for s in self.db.subs(status=("RUNNING",)):
            sid = s["sid"]
            self._bill(s)
            r = self.db.run(sid)
            task, h, st = r["task"], RunHandle(**r["handle"]), r["sup_state"] or {}
            has_upload = self.d.private.last_modified(layout.staging_prefix(sid) + "manifest.json") is not None
            st["live"] = self._sample(sid, st.get("live"))
            try:
                outcome = supervisor.step(self.now(), task, h, self.d.runner, st, has_upload, st["live"].get("done"))
            except Exception as e:  # agent backend / network hiccup: retried next tick, forced stop at the deadline
                log.warning("supervise %s: %s (retrying next tick)", sid, str(e).splitlines()[0][:200])
                if self.now() < task["run"]["deadline"]:
                    continue
                outcome = "finished"
            self.db.save_run(sid, sup_state=st)
            if outcome == "running":
                continue
            if outcome == "lost":
                self._run_lost(s, "the agent backend could not run the agent")
                continue
            exported = False
            try:
                agent_dir = Path(self.d.work_dir) / sid / "agent"
                exported = self.d.runner.export(h, agent_dir)
                if exported:      # published later: without the run's secrets
                    redact.redact_dir(agent_dir, redact.secrets_in_bundle(self._bundle_path(sid)))
            except Exception:
                log.exception("export %s", sid)
                exported = False
            try:
                self.d.fleet.release(sid)
            except Exception:
                log.exception("release the machine of %s (retried by schedule)", sid)
            end = self.now()
            self._bill(s, end=end, agent_usd=st.get("usd"))
            self.db.update(sid, "COLLECTING", record_patch={
                "run": self._run_view(s, {**r, "sup_state": st}, end),     # as status.json showed it, at its end
                "agent_log": "published" if exported else "unavailable",
                "agent_outcome": outcome,
                "fee": {**s["record"].get("fee", {}), "charged_usd": sum(self.db.charged(sid).values()) / 1e6},
                "timeline": {**s["record"]["timeline"], "agent_end": end}})

    def _bill(self, s, end=None, agent_usd=None):
        """A run's GPUs are charged one hour's price at each full hour from kickoff; at its end the GPU time actually
        used is settled (charge 0), the agent's model use is charged (charge -1, at most run.max_agent_usd) and the
        review fee paid on entering review is given back (charge -3): a run costs what it used."""
        rules, kick = self.rules_for(s["sid"]), s["record"]["timeline"]["kickoff"]
        hours = min(int((self.now() - kick) // 3600), int(rules["run"]["hours"]))
        for n in range(1, hours + 1):
            self.db.charge(s["sid"], n, s["coldkey"], pricing.hour_musd(rules))
        if end is not None:
            rest = pricing.used_musd(rules, end - kick) - sum(v for n, v in self.db.charged(s["sid"]).items() if n > 0)
            self.db.charge(s["sid"], 0, s["coldkey"], max(rest, 0))
            self.db.charge(s["sid"], -1, s["coldkey"], pricing.agent_musd(rules, agent_usd))
            fee = self.db.charged(s["sid"]).get(REVIEW_FEE)
            if fee:                       # the run's charges are its cost: the review fee goes back
                self.db.charge(s["sid"], REVIEW_FEE_BACK, s["coldkey"], -fee)

    # ------------------------------------------------------------------------------------------- evaluation machine
    def eval_machine(self):
        """The evaluation machines (eval, eval-2, ...: eval_machines at most, from whatever source the fleet has):
        the first is rented while admission or evaluation work waits and released after eval_idle_minutes without
        any; the others join while more than eval_scale_up models wait (COLLECTING or ADMITTED) and leave, once their
        own job is done, when fewer than eval_scale_down do."""
        f = self.d.fleet
        waiting = len(self.db.subs(status=("COLLECTING", "ADMITTED")))
        if waiting > self.d.eval_scale_up:
            self.db.set("eval_scaled_up", True)
        elif waiting < self.d.eval_scale_down:
            self.db.set("eval_scaled_up", False)
        names = eval_names(self.d.eval_machines)
        want = (len(names) if self.db.get("eval_scaled_up") else 1) if waiting else 0
        busy = {j["machine"] for j in self._eval_jobs().values()}
        for i, name in enumerate(names):
            if f.get(name) is None:
                if i < want:
                    try:
                        f.rent(name, self.d.eval_gpu_name, self.d.eval_gpus)
                    except NoMachine as e:
                        log.info("evaluation machine %s: %s", name, e)
                    except Exception as e:
                        log.warning("renting evaluation machine %s: %s", name, e)
                continue
            if f.vanished(name):
                f.release(name, "vanished")
                continue
            if f.advance(name) != "ready":
                continue
            idle_key = f"eval_idle_since:{name}"
            if i < want or name in busy:
                self.db.set(idle_key, None)
            elif i > 0:
                f.release(name, "evaluation queue short")
            elif not self.db.get(idle_key):
                self.db.set(idle_key, self.now())
            elif self.now() - self.db.get(idle_key) >= self.d.eval_idle_minutes * 60:
                f.release(name, "idle")
                self.db.set(idle_key, None)

    def _eval_ready(self) -> list:
        """The ready evaluation machines, first one first."""
        return [n for n in eval_names(self.d.eval_machines)
                if (r := self.d.fleet.get(n)) is not None and r["state"] == "ready"]

    # ------------------------------------------------------------------------------------------- collect
    def collect(self):
        for s in self.db.subs(status=("COLLECTING",)):
            if not self._due(s):
                continue
            sid, rec = s["sid"], s["record"]
            deadline = self.db.run(sid)["task"]["run"]["deadline"]
            try:
                staging.collect(self.d.private, sid, deadline, Path(self.d.work_dir) / sid / "upload",
                                self.d.comp_rules["upload"])
            except staging.UploadError as e:
                code = "NO_UPLOAD" if "no manifest" in str(e) else "BAD_UPLOAD"
                self._publish_agent(sid, rec)
                self._reject(sid, code, str(e), "FAILED")
                continue
            edit = Path(self.d.work_dir) / sid / "upload" / "edit"
            parent_problem = self._check_parent_ref(edit, rec.get("parent"))
            if parent_problem:
                publisher.publish_dir(self.d.public, layout.submission_edit_prefix(sid), edit)
                publisher.publish_admission(self.d.public, sid, {"passed": False, "violations": [parent_problem],
                                                                  "info": {}, "model_digest": None,
                                                                  "base": rec["base"]})
                self._publish_agent(sid, rec)
                self._reject(sid, "REJECTED_RULES", parent_problem)
                continue
            build_dir = Path(self.d.work_dir) / sid / "build"
            shutil.rmtree(build_dir, ignore_errors=True)
            shutil.copytree(edit, build_dir)
            if rec.get("parent"):     # the published chain under parent/ (not uploaded: it is public already)
                self._materialize_chain(rec["parent"], build_dir / "parent")
            ready = self._eval_ready()
            if not ready:
                continue                       # eval_machine is renting or setting one up
            try:                               # a CPU job: beside an evaluation on that machine
                adm = self.d.executor.admit(ready[0], sid, build_dir,
                                            registry.load_base(rec["base"], self.d.registry_root))
            except Exception as e:
                self._failed(s, "admission", e)
                continue
            publisher.publish_dir(self.d.public, layout.submission_edit_prefix(sid), edit)
            publisher.publish_admission(self.d.public, sid, {k: adm.get(k) for k in
                                                              ("passed", "violations", "info", "model_digest")}
                                        | {"base": rec["base"]})
            self._publish_agent(sid, rec)
            if not adm["passed"]:
                self._reject(sid, "REJECTED_RULES", "; ".join(adm["violations"])[:2000])
                continue
            self.db.update(sid, "ADMITTED", record_patch={
                "model_digest": adm["model_digest"], "timeline": {**rec["timeline"], "admitted": self.now()}})
            self._recovered(s)

    @staticmethod
    def _check_parent_ref(edit_dir: Path, parent_sid):
        """With a parent, the spec says "parent": "parent/edit.json"; without one, it has no parent field."""
        try:
            spec = json.loads((Path(edit_dir) / "edit.json").read_text())
        except ValueError as e:
            return f"edit.json: {e}"
        want = "parent/edit.json" if parent_sid else None
        if spec.get("parent") != want:
            return (f"edit.json parent must be {want!r} (entry.json parent {parent_sid})" if parent_sid
                    else "edit.json names a parent but entry.json declares none")
        return None

    def _materialize_chain(self, sid, dest: Path):
        """Rebuild a published edit with its whole chain: submissions/<sid>/edit/ into dest, its parent's under
        dest/parent/, and so on (each published edit/ under its child's parent/)."""
        dest = Path(dest)
        for rel, data in self._parent_files(sid).items():
            p = dest / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        parent = (self.db.sub(sid) or {}).get("record", {}).get("parent")
        if parent:
            self._materialize_chain(parent, dest / "parent")

    def _publish_agent(self, sid, rec):
        d = Path(self.d.work_dir) / sid / "agent"
        if rec.get("agent_log") == "published" and d.exists():
            publisher.publish_dir(self.d.public, layout.submission_agent_prefix(sid), d)

    # ------------------------------------------------------------------------------------------- failures
    def _backoff(self, s, detail):
        """Our infrastructure failed (host unreachable, provisioning, transfers): never the miner's loss. The submission
        keeps its state and priority and is retried after an exponential backoff (1 min .. 1 h); the operator is
        alerted through the log."""
        n = self.db.get(f"backoff:{s['sid']}:n", 0) + 1
        self.db.set(f"backoff:{s['sid']}:n", n)
        self.db.set(f"backoff:{s['sid']}:until", self.d.clock() + min(3600, 60 * 2 ** (n - 1)))
        self.db.update(s["sid"], reason=detail[:500])
        log.warning("infra problem on %s (attempt %d, retrying with backoff): %s", s["sid"], n, detail[:300])

    def _due(self, s):
        return self.d.clock() >= self.db.get(f"backoff:{s['sid']}:until", 0)

    def _recovered(self, s):
        """The step that was being retried went through: its infrastructure note no longer applies."""
        if self.db.get(f"backoff:{s['sid']}:n", 0):
            self.db.set(f"backoff:{s['sid']}:n", 0)
            self.db.update(s["sid"], reason=None)

    def _failed(self, s, what, e):
        """A step failed: counted against the submission only when its own job failed (JobFailed) and it is a miner's
        submission; anything else is ours (backoff)."""
        if isinstance(e, JobFailed):
            return self._job_failed(s, f"{what}: {e}")
        return self._backoff(s, f"{what}: {e}")

    def _job_failed(self, s, detail):
        """A job on the submission's own model or edit exited non-zero: the submission may be the cause. Counted;
        after run.max_job_failures it fails as EVAL_ERROR (the base model's jobs never count)."""
        n = s["infra_retries"] + 1
        self.db.update(s["sid"], infra_retries=n, reason=detail[:500])
        if n > self.rules_for(s["sid"])["run"]["max_job_failures"]:
            self._reject(s["sid"], "EVAL_ERROR", detail[:2000], "FAILED")

    # ------------------------------------------------------------------------------------------- evaluate
    def evaluate(self):
        """Evaluations run detached on the evaluation machines, one model per machine at a time. Each tick takes in
        the jobs that finished, then gives every idle ready machine the next model: for the ADMITTED candidates in
        priority order, the base model first (once: the reference until a first champion; its results are published
        under models/<base>/base/), then the candidate itself. A model stays on its machine (partial output there is
        resumed after a failure); one whose machine is gone starts again on another. A failure of the base's job or
        of our side backs off (never counted); a failing job of the candidate's own model is counted."""
        jobs = self._eval_jobs()
        for model_id, j in list(jobs.items()):
            self._eval_poll(model_id, j)
        jobs = self._eval_jobs()
        idle = [n for n in self._eval_ready() if n not in {j["machine"] for j in jobs.values()}]
        for s in sorted(self.db.subs(status=("ADMITTED",)), key=self.priority):
            if not self._due(s):
                continue
            rules = self.rules_for(s["sid"])
            todo = [(m, b) for m in self._models(s, rules) if (b := self._missing(m, rules))]
            if not todo:
                self._scored(s)
                continue
            for model_id, benches in todo:
                if model_id in jobs or not idle:
                    continue
                on = idle.pop(0)
                try:
                    job = self.d.executor.start(on, model_id, registry.load_base(rules["base"], self.d.registry_root),
                                                benches, self._salt(rules))
                except Exception as e:
                    self._eval_failed(s, model_id, e)
                    continue
                jobs[model_id] = {"sid": s["sid"], "machine": on, "job": job,
                                  "benchmarks": [f"{b['id']}@{b['version']}" for b in benches]}
                self.db.set("eval_jobs", jobs)
                log.info("evaluating %s on %s", model_id, on)

    @staticmethod
    def _models(s, rules) -> list:
        return [f"base:{rules['base']}", s["sid"]]          # every champion was evaluated as an entry

    def _missing(self, model_id, rules) -> list:
        have = self.db.results(model_id)
        return [registry.load_benchmark(k, self.d.registry_root).doc for k in rules.benchmark_keys() if k not in have]

    def _eval_jobs(self) -> dict:
        return self.db.get("eval_jobs") or {}

    def _eval_poll(self, model_id, j):
        """A detached evaluation job: its results taken in once it finished; dropped when its machine is gone."""
        from ..benchmarks import parse  # noqa: PLC0415
        s, rules = self.db.sub(j["sid"]), self.rules_for(j["sid"])
        row = self.d.fleet.get(j["machine"])
        if row is None or s is None or s["status"] != "ADMITTED":
            return self._eval_drop(model_id)
        if row["state"] != "ready":
            return
        benches = [registry.load_benchmark(k, self.d.registry_root).doc for k in j["benchmarks"]]
        try:
            outs = self.d.executor.result(j["machine"], j["job"], model_id, benches, self._salt(rules))
        except Exception as e:
            self._eval_drop(model_id)
            return self._eval_failed(s, model_id, e)
        if outs is None:
            return
        for key, out in outs.items():
            publisher.publish_results(self.d.public, model_id, key, out)   # first: redone if it fails
            self.db.put_results(model_id, key, parse(out))
        self._eval_drop(model_id)
        if model_id.startswith("base:"):
            self._publish_base_score(model_id, rules)
        if not any(self._missing(m, rules) for m in self._models(s, rules)):
            self._scored(s)

    def _eval_drop(self, model_id):
        jobs = self._eval_jobs()
        jobs.pop(model_id, None)
        self.db.set("eval_jobs", jobs)

    def _eval_failed(self, s, model_id, e):
        if model_id.startswith("base:"):   # the base model's own job: always ours
            return self._backoff(s, f"base evaluation: {e}")
        return self._failed(s, "evaluation", e)

    def _scored(self, s):
        self._recovered(s)
        self.db.update(s["sid"], "SCORED", record_patch={
            "timeline": {**s["record"]["timeline"], "scored": self.now()}})

    def _publish_base_score(self, model_id, rules):
        """The base model's score.json, as an entry's but with no match: its accuracy per benchmark (write-once; a
        later rules version adding benchmarks only adds results files, line.json carries the rest)."""
        res = self.db.results(model_id)
        if not set(rules.benchmark_keys()) <= set(res):
            return
        try:
            publisher.publish_score(self.d.public, model_id, {
                "base": rules["base"], "rules_sha256": rules.sha256, "generated": self._generated(rules),
                "benchmarks": {k: {"acc": round(judge.accuracy(res[k]), 4), "n": len(res[k])}
                               for k in rules.benchmark_keys()}})
        except ImmutableObjectError:
            log.info("%s: score.json already published under earlier rules", model_id)

    # ------------------------------------------------------------------------------------------- decide
    def decide(self):
        """Decisions follow priority (the chain order of each run's committed content), not the order runs finish:
        walking submissions by priority, a SCORED one is decided against the champion of that moment; the walk stops
        at the first earlier submission still in the pipeline (in review, queued, running or being evaluated), so a
        later commitment - e.g. a copy - can never be judged before the content it may have copied. WAITING ones (no
        PR, no funds yet) do not block: nobody can be held up by a miner who does not show up."""
        waiting = ("REVIEW", "QUEUED", "RUNNING", "COLLECTING", "ADMITTED")
        blocked = set()                       # each base's line is decided independently
        for s in sorted(self.db.subs(), key=self.priority):
            if s["base"] in blocked:
                continue
            if s["status"] in waiting:
                blocked.add(s["base"])
            elif s["status"] == "SCORED":
                self._decide_one(s)

    @staticmethod
    def _salt(rules):
        """The sampling salt of a line's evaluations: a model's results are generated once and reused in every match."""
        return f"axonry-{rules['base']}"

    def _generated(self, rules):
        return {"salt": self._salt(rules), "image": self.d.eval_image}

    def _decide_one(self, s):
        """One match: the submission against the champion of this moment only (the base model while its line has no
        champion yet)."""
        sid, rec, base = s["sid"], s["record"], s["base"]
        rules = self.rules_for(sid)
        params = {k: rules.judge_params(k) for k in rules.benchmark_keys()}
        head = self.d.chain.finalized_head()
        champ = self.db.champion(base)
        ref_id = champ["sid"] if champ else f"base:{base}"
        ref = self.db.results(ref_id)
        mine = self.db.results(sid)
        cmp = judge.compare(mine, ref, params, rules["judge"])
        benches = {}
        for k in params:
            acc, c = judge.accuracy(mine[k]), cmp["benchmarks"][k]
            ref_acc = acc - c["d"]                 # paired on the same items: d is the difference of the means
            benches[k] = {"acc": round(acc, 4), "n": len(mine[k]),
                          "vs_champion": {"ref": ref_id, "ref_acc": round(ref_acc, 4), **c, "params": params[k],
                                          # the scores this run had to reach: not worse / a win, on this benchmark
                                          "tie_at": round(ref_acc - max(rules["judge"]["z_tie"] * c["se"],
                                                                        params[k]["eps_tie"]), 4),
                                          "win_at": round(ref_acc + rules["judge"]["z_win"] * c["se"]
                                                          + params[k]["delta_win"], 4)}}
        decision = "NEW_CHAMPION" if cmp["dominates"] else "NOT_DOMINATING"
        publisher.publish_score(self.d.public, sid, {
            "base": base, "rules_sha256": rules.sha256, "model_digest": rec.get("model_digest"),
            "judge": rules["judge"], "generated": self._generated(rules), "benchmarks": benches,
            "wins": cmp["wins"], "tie_all": cmp["tie_all"],
            "dominates_champion": cmp["dominates"], "decision": decision, "decided_block": head})
        self.db.update(sid, "PUBLISHED", reason_code=decision, decided_at=self.d.clock(), record_patch={
            "decision": decision, "wins": cmp["wins"], "reference": ref_id,
            "scores": {k: [b["acc"], b["vs_champion"]["tie_at"], b["vs_champion"]["win_at"],
                           "win" if b["vs_champion"]["win"] else "tie" if b["vs_champion"]["tie_ok"] else "lose"]
                       for k, b in benches.items()},
            "timeline": {**rec["timeline"], "decided": self.now()}})
        self._publish_record(sid)
        if cmp["dominates"]:
            self.db.add_champion(base, sid, s["hotkey"], head, {"reference": ref_id, "wins": cmp["wins"]})
        self._close_pr(sid, f"axonry: submission `{sid}` -> **{decision}** ({cmp['wins']} wins, not worse on all: "
                            f"{cmp['tie_all']}). Details: `axonry-miner board {sid}`.",
                       (rec.get("pr") or {}).get("locked_commit") if cmp["dominates"] else None)
        self._publish_line(base)
        if cmp["dominates"]:
            self.publish_weights()

    def disqualify(self, sid, reason):
        """The owner's public ruling on a run that broke the rules the checks cannot see (e.g. its published agent log
        shows benchmark data or outside weights brought in). It stops counting: a champion's title goes back to the
        latest earlier champion still standing (or the base). Decisions already made against it stand."""
        s = self.db.sub(sid)
        if s is None or s["status"] not in ("SCORED", "PUBLISHED"):
            raise ValueError(f"{sid}: only an evaluated submission can be disqualified")
        was_champion = (self.db.champion(s["base"]) or {}).get("sid") == sid
        self.db.update(sid, "DISQUALIFIED", reason_code="DISQUALIFIED", reason=reason[:2000])
        publisher.publish_ruling(self.d.public, sid, {"decision": "DISQUALIFIED", "reason": reason[:2000],
                                                      "at": self.now(), "was_champion": was_champion})
        self._publish_line(s["base"])
        if was_champion:
            self.publish_weights()

    # ------------------------------------------------------------------------------------------- publication
    def _publish_record(self, sid):
        s = self.db.sub(sid)
        rec = {**s["record"], "coldkey": s["coldkey"], "status": s["status"], "reason_code": s["reason_code"],
               "reason": s["reason"]}
        publisher.publish_submission(self.d.public, sid, rec)

    def _publish_line(self, base):
        """models/<base>/line.json: the base model's line in one file - champion and succession, the base model's own
        accuracy (the first reference), and one row per entry (status, decision, reference and per
        benchmark [accuracy, not-worse-at, win-at, verdict]). Every figure in it also lives in write-once files."""
        rows = []
        champ = self.db.champion(base)
        for s in self.db.subs(base=base):
            r = s["record"]
            rows.append({"sid": s["sid"], "hotkey": s["hotkey"], "status": s["status"], "reason_code": s["reason_code"],
                         "rules": r.get("rules_sha256"), "wins": r.get("wins"), "decision": r.get("decision"),
                         "reference": r.get("reference"), "scores": r.get("scores"),
                         "pr": (r.get("pr") or {}).get("number"), "title": (r.get("pr") or {}).get("title"),
                         "decided_at": s["decided_at"]})
        base_id, ref = f"base:{base}", self.db.results(f"base:{base}")
        publisher.publish_line(self.d.public, base, {
            "champion": ({"sid": champ["sid"], "hotkey": champ["hotkey"], "since_block": champ["block"]}
                         if champ else None),
            "history": [{"seq": c["seq"], "sid": c["sid"], "hotkey": c["hotkey"], "block": c["block"],
                         "evidence": json.loads(c["evidence"])} for c in self.db.champions(base)],
            "base_model": {"id": base_id, "accuracy": {k: round(judge.accuracy(v), 4) for k, v in ref.items()} or None},
            "entries": rows})

    def _in_force(self, head):
        """The rules in force at `head` (a next version may be staged), or None before the first takes effect."""
        try:
            return registry.rules_at(head, self.d.registry_root)
        except registry.RegistryError:
            return None

    def publish_weights(self):
        head = self.d.chain.finalized_head()
        rules = self._in_force(head)
        if rules is None:
            return
        base = rules["base"]
        champ = self.db.champion(base)
        info = None
        if champ:
            uid = self.d.chain.uid_of(champ["hotkey"], rules["netuid"])
            info = {"sid": champ["sid"], "hotkey": champ["hotkey"], "uid": uid}
        published = json.loads(self.d.public.get(layout.WEIGHTS) or "{}").get("sequence", 0)
        seq = max(self.db.get("weights_sequence", 0), published) + 1     # a restored state DB may lag behind
        doc = publisher.build_weights(rules["netuid"], base, info, seq, head, burn=self.d.burn)
        publisher.publish_weights(self.d.public, doc, self.d.owner_keypair)
        self.db.set("weights_sequence", seq)
        publisher.publish_index(self.d.public, rules, self.d.registry_root)

    def bootstrap(self):
        """Publish definitions, every base model's line, the index of the rules in force, and a first weights.json
        (uid0 = burn) if none exists yet."""
        publisher.publish_registry(self.d.public, self.d.registry_root)
        for base in sorted({r["base"] for r in registry.all_rules(self.d.registry_root)}):
            self._publish_line(base)
        if self.d.public.get(layout.WEIGHTS) is None:
            self.publish_weights()
        elif rules := self._in_force(self.d.chain.finalized_head()):     # the rules in force, as now in the registry
            publisher.publish_index(self.d.public, rules, self.d.registry_root)
        self._reopen_refused()

    def _reopen_refused(self):
        """Once: entries a review refused when that still ended them (never run) may now be fixed by deletions like
        any other, their window starting now. Their content stays refused; the record they were ended with is
        withdrawn (a submission's record is written once, when it ends)."""
        if self.db.get("reopened_refused"):
            return
        for s in self.db.subs(status=("REJECTED",)):
            if s["reason_code"] != "REVIEW_CHANGES" or s["locked_commit"]:
                continue
            rec, sid = s["record"], s["sid"]
            hours = None if s["hotkey"] == self.d.owner_keypair.ss58_address \
                else self.rules_for(sid).doc.get("reveal_hours")
            self.d.public.delete_prefix(layout.submission(sid, "submission.json"))
            self.db.update(sid, "WAITING", decided_at=None, record_patch={
                "pr_result": None, "refused": {self._content(s): (s["reason"] or "")[:600]},
                "fix_by": int(self.now() + float(hours) * 3600) if hours is not None else 0})
            log.info("%s: refused by review; may now be fixed by deletions", sid)
        self.db.set("reopened_refused", True)

    def refresh_weights(self, max_age=86400):
        """Re-issue weights.json when it is old, another base model's line is in force, the owner's burned share
        changed, or the champion's uid changed (re-registration). index.json follows the rules version in force."""
        raw = self.d.public.get(layout.WEIGHTS)
        if raw is None:
            return self.publish_weights()
        cur = json.loads(raw)
        champ = cur.get("champion")
        moved = bool(champ) and self.d.chain.uid_of(champ["hotkey"], cur["netuid"]) != champ.get("uid")
        rules = self._in_force(self.d.chain.finalized_head())
        new_base = rules is not None and rules["base"] != cur.get("base")
        burn = cur.get("burn", 0.0) != self.d.burn             # the owner changed the burned share
        if moved or new_base or burn or self.d.clock() - cur.get("issued_at", 0) > max_age:
            self.publish_weights()
        elif rules is not None and (json.loads(self.d.public.get(layout.INDEX) or "{}").get("rules")
                                    != layout.rules_version(rules.sha256)):
            publisher.publish_index(self.d.public, rules, self.d.registry_root)    # a new rules version took effect

    def publish_status(self, every=60):
        """status.json: everything in priority order - the review queue (reviewers take it top-down), the run queue,
        runs in progress and evaluations; submissions waiting (and why) and those stopped before a run; PRs that carry
        no submission (and why); every account (deposits, charges, balance per asset); every registered miner with its
        balance and its run in progress (the frontend's miner view)."""
        if self.d.clock() - self.db.get("status_at", 0) < every:
            return
        def row(s):
            pr = s["record"].get("pr") or {}
            starting = s["status"] == "QUEUED" and self.d.fleet.get(s["sid"]) is not None   # its machine sets up
            return {"sid": s["sid"], "hotkey": s["hotkey"], "pr": s["pr_number"],
                    "title": pr.get("title"),
                    "status": "STARTING" if starting else s["status"],
                    "reason_code": None if starting else s["reason_code"], "reason": None if starting else s["reason"],
                    "commit": s["locked_commit"] or s["head_sha"], "content_digest": self._content(s),
                    "directory": pr.get("directory"),
                    "priority": [s["prio_block"], s["prio_index"]],
                    **({"revealed": self._revealed(s)} if self._content(s) != s["content_digest"] else {})}
        by_prio = sorted(self.db.subs(), key=self.priority)
        rules = self._in_force(self.d.chain.finalized_head())
        if rules is None:
            return
        champ = self.db.champion(rules["base"])
        accounts = self.db.accounts()
        doc = {"schema": publisher.SCHEMAS["status"], "updated_at": self.now(), "base": rules["base"],
               "repo": getattr(self.d.github, "repo", None),
               "champion": self._champion_row(champ),
               "miners": self._miners(rules["netuid"], accounts),
               "review": [row(s) for s in by_prio if s["status"] == "REVIEW"],
               "queue": [{**row(s), "position": i + 1}
                         for i, s in enumerate(x for x in by_prio if x["status"] == "QUEUED")],
               "waiting": [row(s) for s in by_prio if s["status"] == "WAITING"],
               "stopped": [row(s) for s in by_prio if s["status"] in ("REJECTED", "FAILED") and not s["locked_commit"]],
               "running": [row(s) for s in by_prio if s["status"] == "RUNNING"],
               "evaluating": [row(s) for s in by_prio if s["status"] in ("COLLECTING", "ADMITTED", "SCORED")],
               "prs": sorted(self.db.get("prs", {}).values(), key=lambda w: w["pr"]),
               "contents": self._contents(),
               "accounts": accounts}
        self.d.public.put(layout.STATUS, pretty(doc))
        self.db.set("status_at", self.d.clock())

    def _champion_row(self, champ):
        if not champ:
            return None
        rec = (self.db.sub(champ["sid"]) or {}).get("record", {})
        pr = rec.get("pr") or {}
        return {"sid": champ["sid"], "hotkey": champ["hotkey"], "pr": pr.get("number"),
                "title": pr.get("title")}

    def _sample(self, sid, last=None) -> dict:
        """The run's machine now: GPU use and the agent's one-line status (secrets of the run removed). What a sample
        cannot read (a slow or sick host) keeps the last reading."""
        last = last or {}
        try:
            live = self.d.fleet.sample(sid)
            if live["gpus"] is None:
                live["gpus"] = last.get("gpus")
            if live["note"]:
                live["note"] = redact.redact(live["note"], redact.secrets_in_bundle(self._bundle_path(sid)))
        except Exception as e:   # the run goes on
            log.warning("sample %s: %s", sid, str(e).splitlines()[0][:200])
            live = {k: last[k] for k in ("gpus", "note", "done") if k in last}
        return {**live, "at": self.now()}

    def _run_view(self, s, r, at):
        """A run as published: live in status.json while it runs, and kept in its record when it ends."""
        sup, handle = r.get("sup_state") or {}, (r.get("handle") or {}).get("data") or {}
        live, kick = sup.get("live") or {}, s["record"]["timeline"].get("kickoff")
        return {"sid": s["sid"], "deadline": (r.get("task") or {}).get("run", {}).get("deadline"),
                "note": live.get("note"), "gpus": live.get("gpus"), "sampled_at": live.get("at"),
                "agent": {"runner": getattr(self.d.runner, "name", None),
                          "model": handle.get("model") or getattr(self.d.runner, "model", None),
                          "tokens_m": round(sup["tokens"] / 1e6, 2) if sup.get("tokens") is not None else None,
                          "usd": round(sup["usd"], 2) if sup.get("usd") is not None else None},
                "gpu_usd": (round(pricing.used_musd(self.rules_for(s["sid"]), at - kick) / pricing.MUSD, 2)
                            if kick else None)}

    def _miners(self, netuid, accounts):
        """Who is registered right now, with what a miner's page shows: [{uid, hotkey, coldkey, entry (the sid of its
        one entry, None before it commits), account (its coldkey's balance), run (its run in progress: deadline, the
        agent's one-line status, each GPU's use, sampled_at; None when it has none)}] (owners cached; a hotkey's
        owner does not change)."""
        owners = self.__dict__.setdefault("_owners", {})
        runs = {}
        for s in self.db.subs(status=("RUNNING",)):
            runs[s["hotkey"]] = self._run_view(s, self.db.run(s["sid"]) or {}, self.now())
        out = []
        for uid, hk in sorted(self.d.chain.uid_hotkeys(netuid).items()):
            if hk not in owners:
                owners[hk] = self.d.chain.owner_of(hk, None)
            entry = self.db.of_hotkey(hk)
            out.append({"uid": uid, "hotkey": hk, "coldkey": owners[hk], "entry": entry and entry["sid"],
                        "account": accounts.get(owners[hk]), "run": runs.get(hk)})
        return out

    def _contents(self):
        """Every submission still in progress whose content a PR has made public (its PR is open), in priority order:
        a reviewer compares an item with every earlier one of these (originality, review/PROTOCOL.md). A finished
        entry's method is free to build on."""
        return [{"sid": s["sid"], "hotkey": s["hotkey"], "pr": s["pr_number"], "status": s["status"],
                 "commit": s["locked_commit"] or s["head_sha"],
                 "directory": (s["record"].get("pr") or {}).get("directory"),
                 "priority": [s["prio_block"], s["prio_index"]]}
                for s in sorted(self.db.subs(), key=self.priority)
                if s["pr_number"] and s["status"] not in self.FINAL]

    def janitor(self, every=3600, keep_snapshots=8):
        """Hourly: what finished submissions leave behind - their staging upload and local work directory (task bundle
        with the run's keys, agent log, build; all that matters is published) - and old state snapshots."""
        if self.d.clock() - self.db.get("janitor_at", 0) < every:
            return
        for s in self.db.subs(status=("PUBLISHED", "REJECTED", "FAILED")):
            self.d.private.delete_prefix(layout.staging_prefix(s["sid"]))
            shutil.rmtree(Path(self.d.work_dir) / s["sid"], ignore_errors=True)
        snaps = sorted(k for k, _ in self.d.private.list(layout.STATE_PREFIX))
        for k in snaps[:-keep_snapshots]:
            self.d.private.delete_prefix(k)
        self.db.set("janitor_at", self.d.clock())

    ROLES = {
        # fast, time-critical: deadlines, machines and the queue must never wait for a benchmark run
        "main": ("intake", "schedule", "supervise", "decide", "settle_prs", "refresh_weights", "publish_status",
                 "janitor", "snapshot_state", "keep_agents"),
        # slow GPU work on the evaluation machine (admission builds, benchmark generation): its own process
        "eval": ("eval_machine", "collect", "evaluate"),
    }

    def tick(self, role="all"):
        names = self.ROLES["main"] + self.ROLES["eval"] if role == "all" else self.ROLES[role]
        for name in names:
            try:
                getattr(self, name)()
            except Exception:
                log.exception("tick %s", name)

    def keep_agents(self):
        self.d.runner.keep_alive()

    def snapshot_state(self):
        """A consistent copy of the state DB to the private bucket every 6 hours (restore: axonry-orchestrator
        restore-state)."""
        if self.now() - (self.db.get("snapshot_at") or 0) < 6 * 3600:
            return
        self.d.private.put(layout.STATE_PREFIX + f"db-{self.now()}.sqlite", self.db.snapshot(),
                           "application/octet-stream")
        self.db.set("snapshot_at", self.now())

