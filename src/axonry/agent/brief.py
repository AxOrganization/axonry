"""Everything the agent must know, generated: BRIEF.md for reading and task.json["limits"] for tools.

Sources, never hand-written: the resources probed in the sandbox, the run's clock, the competition repo's
rules.json (edit limits) and the subnet rules version (time, upload protocol). Change a rule and the brief follows.
"""
import json
import time


def limits_from(comp_rules: dict) -> dict:
    """The subset of the competition protocol (WeightSpaceProgramming rules.json, section "edit") that binds this run."""
    from ..evaluation.admit_remote import edit_limits  # noqa: PLC0415
    e = comp_rules["edit"]
    t = edit_limits(comp_rules)
    return {
        "editable_tensors": e["editable_tensors"],
        "ops_allowed": e["ops"],
        "payload": e["payload"],
        "relative_change": e["relative_change"],
        "max_payload_bytes": t["max_payload_bytes"],
        "max_relative_change_per_tensor": t["max_relative_change_per_tensor"],
        "max_global_relative_change": t["max_global_relative_change"],
        "upload_files": comp_rules["upload"]["files"],
        "no_overfitting": e["no_overfitting"],
    }


def _parent(task) -> str:
    p = task.get("parent")
    if not p:
        return ""
    return f"""
## Starting point: the parent edit
This entry stacks on the published submission `{p['sid']}`: its edit (with its own parents) is in this bundle under
`parent/edit/`. Copy it to the sandbox as `parent/edit/` next to task.json (where `axonry-check` looks for it) and
start the case from it (`python -m wsp.case init <case> ... --parent parent/edit/edit.json`). Your edit is a new
layer on that model: keep its `"parent"` (wsp.upload writes it as `parent/edit.json`); the limits apply to your own
ops, measured against the parent model.
"""


def _judged(task) -> str:
    j = task.get("judge")
    if not j:
        return ""
    if j["champion"]:
        ref = f"the champion `{j['champion']}`"
    else:
        ref = "the base model (no champion yet)"
    return f"""
## How the edit is judged
Against {ref}, on the benchmarks: the edit must be worse on none of them (beyond their noise) and clearly better on at least {j['n_win']}.
"""


def _fmt_ts(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts))


def render(task: dict) -> str:
    lim, res, run = task["limits"], task["sandbox"]["resources"], task["run"]
    budget = (f"\n- Your agent budget: {run['max_agent_usd']} USD for model use; at 80% you are asked to upload, at 100% the "
              "run stops." if run.get("max_agent_usd") else "")
    gpus = ", ".join(f"{g['name']} ({g['memory_gib']} GiB)" for g in res.get("gpus", [])) or "none detected"
    return f"""# Task brief — submission {task['sid']}

This brief is generated from the competition rules; it is the authoritative list of what you have and what you may
do. Your method is in `submission/METHOD.md` (written by the miner); follow it within these limits. `submission/` is
the method exactly as reviewed: work from it, the toolkit and practice data only, and never look up its pull request,
earlier versions of it, or anything else about it or the evaluation benchmarks elsewhere.

## Goal
Improve base model `{task['base']}` on the benchmarks by **editing its weights directly** (method guide: the
toolkit's weight-edit skill: `.claude/skills/weight-edit/` (Claude-style agents) or `.agents/skills/weight-edit/`
(Codex), same content, under `/workspace/WeightSpaceProgramming/`; toolkit README alongside), producing an edit spec
(`edit.json` + `edit.npz`) that the organizer applies to the official base weights{" on top of the parent edit" if task.get("parent") else ""}.
You do not upload weights. Benchmarks are not available to you; use the practice data only. Carry out the method
yourself: the edits already in the toolkit (its `cases/`) are other people's results, and uploading one of them, or
the parent unchanged, improves nothing. How the edit is found and chosen is the method's and the skill's, never your
own rule: choices on dev sets, holdout only to confirm a frozen edit.
{_parent(task)}{_judged(task)}
## Time
- Start: {_fmt_ts(run['start'])}
- **Deadline: {_fmt_ts(run['deadline'])}** ({(run['deadline'] - run['start']) // 3600} h). Anything uploaded later is ignored.
- Reminders {' and '.join(f"{m} min" for m in run['wrap_minutes'])} before the deadline: then stop all other work and upload
  your best result. `axonry-time` in the sandbox prints the time left.
- One background job of yours lasts at most 60 minutes; you are then asked to continue from the sandbox's PROGRESS.md.
  Nobody answers questions during the run: carry on with what is left yourself, never ask whether to. Run long GPU
  work detached in the sandbox (`tmux` or `nohup`), so it keeps running when a job ends, and check on it later.{budget}
- Upload an intermediate result early and re-upload improvements: the last complete upload before the deadline counts.
  An upload still in progress at the deadline does not count, and may spoil the one before it.
- Finished early (a complete result uploaded, nothing left worth trying)? Run `axonry-done` in the sandbox: the run
  ends then, and its GPU time is charged only up to that moment.

## Machine (one sandbox, exclusive to you until the deadline)
- Connect once with `bash ssh/install.sh` (from this bundle), then always `ssh -F ~/.axonry/ssh/config sandbox`
  (also `scp -F ~/.axonry/ssh/config file sandbox:/workspace/work/`, `rsync -e "ssh -F ~/.axonry/ssh/config"`).
  Host keys are pinned; use `tmux` inside the sandbox for long jobs so a dropped connection does not kill them.
- GPU: {gpus}; CPUs: {res.get('cpus')}; memory: {res.get('memory')}; free disk: {res.get('disk_gib')} GiB.
- Base model (read-only): `{res.get('base_path')}`; competition toolkit (read-only): `{res.get('competition_repo')}`;
  your writable space: `{res.get('work_path')}`.
- Network: {res.get('network')}. Never download models, weights or benchmark data: everything you need is already
  there, and your public agent log shows every command.
- On your own (agent) machine, work only inside your workspace folder: do not use `/tmp`, `/dev/shm` or any other
  directory outside it.
- The sandbox is destroyed at the deadline.

## Rules for the edit (violations are rejected before evaluation, with the reason published)
- Editable: {lim['editable_tensors']}
- Ops: {', '.join(lim['ops_allowed'])} (declarative, applied to the starting model's weights).
- Payload (arrays in the spec) at most {lim['max_payload_bytes']} bytes; any number of tensors.
- Relative change at most {lim['max_relative_change_per_tensor']} per tensor and {lim['max_global_relative_change']} over all editable matrices.
- Payload: {lim['payload']}.
- Relative change: {lim['relative_change']}.
- No overfitting: {lim['no_overfitting']}
- Practice data ships with the toolkit (`practice:` sets in wsp.data); benchmark sets are not here.
- Self-check before uploading, in the sandbox: `axonry-check <dir with edit.json, edit.npz, REPORT.md> --task task.json`
  runs exactly the organizer's admission check (same code).

## Upload (protocol {task['upload_protocol']}, defined in the toolkit's rules.json)
1. `python -m wsp.upload <case> <edit> --task task.json` (in the sandbox: copy task.json there first)
   packages the case's edit and report and uploads the files, then `manifest.json`, to `{task['r2']['prefix']}`
   with the temporary credentials in `task.json` (they expire at the deadline). Cases live in `$WSP_CASES`.
2. Files: {', '.join(lim['upload_files'])}; `edit.json` must have `"base": "{task['base']}"`.
3. `REPORT.md`: what failure you targeted, how the spec was derived, controls, what did not work.

## Progress
Keep `PROGRESS.md` in the sandbox, at `/workspace/work/PROGRESS.md`, up to date: what is done, what is next. Its
first line is your status, shown publicly while you run: keep it current as you progress, one plain sentence of at
most 20 words, readable by anyone following the run.
When a turn ends before the deadline you will be asked to continue from it.
"""


def ssh_files(task: dict) -> dict:
    """ssh/ of the bundle: key, pinned known_hosts and a config -> `ssh -F ~/.axonry/ssh/config sandbox`. A sandbox
    behind a jump host (an owned host's container) is reached through that host's sshd, with the same key."""
    sb, jump = task["sandbox"], task["sandbox"].get("jump")
    d = "@AXONRY_SSH_DIR@"   # install.sh writes the absolute path ($HOME/.axonry/ssh): no reliance on ssh's %d / ~
    keys = (f"  IdentityFile {d}/id_sandbox\n  IdentitiesOnly yes\n"
            f"  UserKnownHostsFile {d}/known_hosts\n  StrictHostKeyChecking yes\n")
    config = f"# sandbox of submission {task['sid']}, valid until the deadline\n"
    if jump:
        config += (f"Host axonry-jump\n  HostName {jump['host']}\n  Port {jump['port']}\n  User {jump['user']}\n"
                   f"  HostKeyAlias axonry-jump\n{keys}")
    config += (f"Host sandbox\n  HostName {sb['host']}\n  Port {sb['port']}\n  User {sb['user']}\n"
               f"  HostKeyAlias sandbox\n{keys}  ServerAliveInterval 30\n" + ("  ProxyJump axonry-jump\n" if jump else ""))
    known = f"sandbox {sb['host_key']}\n" + (f"axonry-jump {jump['host_key']}\n" if jump else "")
    install = ("#!/bin/sh\n# install this run's ssh access: ssh -F ~/.axonry/ssh/config sandbox\nset -e\numask 077\n"
               'here=$(cd "$(dirname "$0")" && pwd)\nmkdir -p ~/.axonry/ssh && chmod 700 ~/.axonry ~/.axonry/ssh\n'
               'cp "$here/known_hosts" "$here/id_sandbox" ~/.axonry/ssh/\n'
               'sed "s#@AXONRY_SSH_DIR@#$HOME/.axonry/ssh#g" "$here/config" > ~/.axonry/ssh/config\nchmod 600 ~/.axonry/ssh/*\n'
               "ssh -F ~/.axonry/ssh/config sandbox true && echo 'sandbox reachable: ssh -F ~/.axonry/ssh/config sandbox'\n")
    key = sb["private_key"] if sb["private_key"].endswith("\n") else sb["private_key"] + "\n"
    return {"config": config.encode(), "known_hosts": known.encode(),
            "install.sh": install.encode(), "id_sandbox": key.encode()}


def kickoff_message(bundle_name: str) -> str:
    return (f"The attached {bundle_name} contains task.json, "
            "BRIEF.md and submission/. Unpack it, read BRIEF.md first (it is authoritative), then "
            "submission/METHOD.md, and carry out the method on the GPU sandbox described in task.json. Upload results "
            "as BRIEF.md says; upload an early result and improve it. Keep the sandbox's /workspace/work/PROGRESS.md current.")


def continue_message(task: dict, now: int) -> str:
    left = max(task["run"]["deadline"] - now, 0)
    return (f"Continue the task (submission {task['sid']}) from the sandbox's PROGRESS.md. Time left: "
            f"{left // 3600} h {left % 3600 // 60} min until {_fmt_ts(task['run']['deadline'])}. "
            "Use it to carry the method further and improve your result (a status report is not progress); re-read "
            "BRIEF.md if needed, and make sure a complete result is uploaded before the deadline.")


def wrap_message(task: dict, now: int) -> str:
    left = max(task["run"]["deadline"] - now, 0)
    return (f"{left // 60} minutes left until the deadline. Stop all other work now: run axonry-check (--task task.json) "
            "on your best result and upload it (python -m wsp.upload ... --task task.json) with an up-to-date REPORT.md. "
            "An upload still in progress at the deadline does not count.")


def budget_message(task: dict, usd: float) -> str:
    return (f"This run's agent has used {usd:.2f} of its {task['run']['max_agent_usd']} USD. Stop exploring: run "
            "axonry-check (--task task.json) on your best result and upload it now (python -m wsp.upload ... --task task.json); the run "
            "stops when the budget is used up.")


def task_json(task: dict) -> bytes:
    return json.dumps(task, indent=1, sort_keys=True).encode()
