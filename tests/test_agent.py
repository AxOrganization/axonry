"""Run supervision against a fake agent backend, and the arbos chat-state reading (shapes seen in real runs)."""
import json

from axonry.agent import supervisor
from axonry.agent.arbos import ArbosError, ArbosRunner, busy, project_name
from axonry.agent.base import RunHandle, RunStatus

T0 = 1_800_000_000
TASK = {"sid": "s1", "run": {"start": T0, "deadline": T0 + 6 * 3600, "wrap_minutes": [30, 10], "stall_minutes": 20,
                             "max_agent_usd": 10}}


class Backend:
    def __init__(self):
        self.status, self.sent, self.stopped = RunStatus(True, T0), [], False

    def poll(self, h):
        return self.status

    def steer(self, h, message):
        self.sent.append(message)

    def stop(self, h):
        self.stopped = True

    def reject(self, h, a, reason):
        pass


def row(status, job=None, role="assistant", ts=0):
    return {"role": role, "status": status, "createdAt": ts, **({"job": {"id": job}} if job else {})}


def test_a_job_that_hit_the_60_minute_cap_is_not_work():
    """Seen in both trials: the job's row ends in error, then a streaming row of the same job lingers."""
    doc = {"messages": [row("done", role="user"), row("error", job="J1"), row("done"), row("streaming", job="J1")]}
    assert not busy(doc)
    doc["messages"].append(row("streaming", job="J2"))                # another job really is working
    assert busy(doc)


def test_the_agent_cost_cap():
    b, st = Backend(), {}
    b.status = RunStatus(True, T0, usd=7.9)
    assert supervisor.step(T0 + 600, TASK, None, b, st, False) == "running" and b.sent == []
    b.status = RunStatus(True, T0 + 600, usd=8.0)                      # 80%: one reminder to upload
    assert supervisor.step(T0 + 660, TASK, None, b, st, False) == "running" and "upload" in b.sent[0]
    assert supervisor.step(T0 + 720, TASK, None, b, st, False) == "running" and len(b.sent) == 1
    b.status = RunStatus(True, T0 + 700, usd=10.0)                     # 100%: stopped, collected like a deadline
    assert supervisor.step(T0 + 780, TASK, None, b, st, False) == "budget" and b.stopped and st["usd"] == 10.0


def test_no_cost_report_means_no_cap():
    b, st = Backend(), {}
    b.status = RunStatus(True, T0, usd=None)                          # arbos could not be read just now
    assert supervisor.step(T0 + 600, TASK, None, b, st, False) == "running" and "usd" not in st


def test_a_backend_that_cannot_run_the_agent_loses_the_run():
    b, st = Backend(), {}
    b.status = RunStatus(False, T0, error="arbos account main: box held")
    assert supervisor.step(T0 + 60, TASK, None, b, st, False) == "running" and b.sent == []
    b.status = RunStatus(True, T0)                                     # it recovered: the clock restarts
    assert supervisor.step(T0 + 120, TASK, None, b, st, False) == "running" and "error_since" not in st
    b.status = RunStatus(False, T0, error="held")
    supervisor.step(T0 + 180, TASK, None, b, st, False)
    assert supervisor.step(T0 + 180 + 20 * 60, TASK, None, b, st, False) == "lost" and b.stopped


def test_reminders_before_the_deadline_and_finishing_when_the_agent_says_so():
    b, st, dl = Backend(), {}, TASK["run"]["deadline"]
    b.status = RunStatus(True, dl)                                    # busy, active just now
    assert supervisor.step(dl - 31 * 60, TASK, None, b, st, True) == "running" and b.sent == []
    supervisor.step(dl - 30 * 60, TASK, None, b, st, True)
    supervisor.step(dl - 20 * 60, TASK, None, b, st, True)
    supervisor.step(dl - 10 * 60, TASK, None, b, st, True)
    assert len(b.sent) == 2 and all("Stop all other work" in m for m in b.sent)       # 30 and 10 minutes before
    b.status = RunStatus(False, T0)                                   # idle with an upload: it goes on
    assert supervisor.step(dl - 9 * 60, TASK, None, b, st, True) == "running" and not b.stopped
    assert supervisor.step(dl - 7 * 60, TASK, None, b, st, True, done=True) == "finished" and b.stopped
    b2 = Backend()
    b2.status = RunStatus(False, T0)
    assert supervisor.step(dl - 7 * 60, TASK, None, b2, {}, False, done=True) == "finished" and b2.stopped  # none


def test_the_agent_cost_is_its_projects_use_since_kickoff():
    use = {"usd": 3.0, "input": 100, "output": 20, "cache_read": 1000, "cache_write": 80}

    def run(account, args, timeout):
        return 0, json.dumps({"all": dict(use), "today": dict(use)}), ""
    r = ArbosRunner({"a": 1}, runner=run)
    h = RunHandle("s1", "p", "c", {"account": "a"})
    r.kickoff(h, "go", "bundle.tgz")                                   # a run lost before used 3.0 of this project
    use.update(usd=7.5, input=300, output=70)
    assert r.cost(h) == 4.5 and r.use(h) == (4.5, 250)                           # tokens: also since kickoff


def test_a_held_arbos_account_takes_no_run():
    def run(account, args, timeout):
        if args[1:3] == ["machine", "status"]:
            return 0, json.dumps({"held": account == "a", "spend_today_usd": 1, "spend_cap_usd": 20}), ""
        return 0, json.dumps([]), ""
    r = ArbosRunner({"a": 3, "b": 1}, runner=run)
    assert r.pick([]) == "b"                                           # a has more room but is held
    try:
        r.pick([RunHandle("x", "p", "c", {"account": "b"})])
        raise AssertionError("expected no room")
    except ArbosError:
        pass


def test_keep_alive_refreshes_every_account():
    done = []

    def refresh(account):
        if account == "a":
            raise RuntimeError("relay down")
        done.append(account)
    ArbosRunner({"a": 1, "b": 1, "c": 1}, runner=lambda *a: (0, "", ""), refresh=refresh).keep_alive()
    assert done == ["b", "c"]


def test_each_run_sets_the_project_models(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)

    def run(account, args, timeout):
        if args[1:3] == ["machine", "status"]:
            return 0, json.dumps({}), ""
        if args[1:3] == ["project", "list"]:
            return 0, json.dumps([{"name": project_name("hk"), "id": "P1"}]), ""
        return 0, json.dumps({"id": "C1"}), ""
    set_to = []
    models = {"subagent_model": {"id": "anthropic/claude-opus-5.5", "effort": "high"}}
    r = ArbosRunner({"a": 1}, runner=run, models=models, set_models=lambda *a: set_to.append(a))
    h = r.prepare("s1", "hk")
    assert (h.workspace, h.chat) == ("P1", "C1") and set_to == [("a", "P1", models)]
