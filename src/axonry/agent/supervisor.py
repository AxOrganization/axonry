"""Supervision of one run. `step` is called every poll interval; it is pure apart from the
runner calls, and keeps its memory in `state` (persisted by the orchestrator), so it survives restarts.

- approval cards are always rejected (nobody is there to approve);
- a finished turn before the deadline gets a "continue" (arbos caps one background job at 60 minutes);
- no activity for stall_minutes: one nudge, then end early;
- a reminder at each of run.wrap_minutes before the deadline (stop working, upload the result now); stopped at the
  deadline;
- the agent finishes early by saying so (`axonry-done` in the sandbox) once a complete upload exists;
- its model use (run.max_agent_usd): a reminder to upload at 80%, stopped at 100%;
- the backend cannot run it (st.error, e.g. the arbos box is held) for stall_minutes: "lost" (ours: the run is
  started again, nothing charged).
"""
from . import brief

CONTINUE_MIN_GAP = 300


def step(now: int, task: dict, handle, runner, state: dict, has_upload: bool, done: bool = False, polled=None) -> str:
    """-> "running" | "finished" (deadline, or the agent said it is done) | "stalled" | "budget" (cost cap) | "lost".
    polled: the agent's status read beforehand (or the exception reading it raised); None: read it here."""
    run = task["run"]
    if now >= run["deadline"]:
        runner.stop(handle)
        return "finished"
    if isinstance(polled, Exception):
        raise polled
    st = polled or runner.poll(handle)
    if st.error:
        state.setdefault("error_since", now)
        if now - state["error_since"] >= run["stall_minutes"] * 60:
            runner.stop(handle)
            return "lost"
        return "running"
    state.pop("error_since", None)
    if done and has_upload:
        runner.stop(handle)
        return "finished"
    cap = run.get("max_agent_usd")
    if st.tokens is not None:
        state["tokens"] = st.tokens
    if st.usd is not None:
        state["usd"] = st.usd
        if cap and st.usd >= cap:
            runner.stop(handle)
            return "budget"
        if cap and st.usd >= 0.8 * cap and not state.get("budget_warned"):
            runner.steer(handle, brief.budget_message(task, st.usd))
            state["budget_warned"] = True
            return "running"
    for a in st.pending_approvals:
        runner.reject(handle, a, "Unattended run: approvals are not available. Continue without this action.")
    due = [m for m in run["wrap_minutes"] if now >= run["deadline"] - m * 60 and m not in state.get("wrapped", [])]
    if due:
        runner.steer(handle, brief.wrap_message(task, now))
        state["wrapped"] = state.get("wrapped", []) + due
        return "running"
    stall = run["stall_minutes"] * 60
    if st.busy:
        if now - st.last_activity >= stall:
            if state.get("stall_nudged_at") and now - state["stall_nudged_at"] >= stall:
                runner.stop(handle)
                return "stalled"
            if not state.get("stall_nudged_at"):
                runner.steer(handle, brief.continue_message(task, now))
                state["stall_nudged_at"] = now
        else:
            state.pop("stall_nudged_at", None)
        return "running"
    state.pop("stall_nudged_at", None)
    if now - state.get("last_continue", 0) >= CONTINUE_MIN_GAP:
        runner.steer(handle, brief.continue_message(task, now))
        state["last_continue"] = now
    return "running"
