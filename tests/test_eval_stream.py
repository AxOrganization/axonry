"""The evaluation stream (eval_remote): item-level scheduling over all benchmarks, resume, completion."""
import json
import threading
import time

from axonry.evaluation.eval_remote import requests_for, run_stream

BENCHES = [{"key": "a@1", "task": "bench:a", "n": 2, "limit": 3}, {"key": "b@1", "task": "bench:b", "n": 1}]
ITEMS = {"bench:a": [{"id": f"a{i}"} for i in range(5)], "bench:b": [{"id": f"b{i}"} for i in range(4)]}


def reqs():
    return requests_for(BENCHES, ITEMS.__getitem__, lambda r: f"Q {r['id']}", lambda i, k, s: hash((i, k, s)), "salt")


def grade(r, text, finish):
    return {"correct": text.endswith("ok")}


def test_requests_follow_benchmark_order_with_limit_and_samples():
    got = [(q[0], q[1]["id"], q[2]) for q in reqs()]
    assert got[:2] == [("a@1", "a0", 0), ("a@1", "a0", 1)] and len(got) == 3 * 2 + 4 and got[-1] == ("b@1", "b3", 0)


def test_slots_stay_full_and_a_long_answer_blocks_nobody(tmp_path):
    lock, inflight, peak, order = threading.Lock(), [0], [0], []

    def send(prompt, seed):
        with lock:
            inflight[0] += 1
            peak[0] = max(peak[0], inflight[0])
        time.sleep(0.5 if prompt == "Q a0" else 0.01)      # one long-tail answer
        with lock:
            inflight[0] -= 1
            order.append(prompt)
        return prompt + " ok", "stop", 3

    run_stream(reqs(), send, grade, tmp_path, concurrency=3)
    assert peak[0] == 3
    assert order[-1] == "Q a0" and "Q b3" in order[:-2]       # benchmark b finished while a0 was still generating
    rows = [json.loads(x) for x in (tmp_path / "b@1.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and all(r["correct"] for r in rows)
    assert (tmp_path / "a@1.jsonl").exists() and not (tmp_path / "a@1.part.jsonl").exists()


def test_a_failed_run_resumes_where_it_stopped(tmp_path):
    sent = []

    def flaky(prompt, seed):
        sent.append(prompt)
        if prompt == "Q b2":
            raise RuntimeError("server gone")
        return "x ok", "stop", 1

    try:
        run_stream(reqs(), flaky, grade, tmp_path, concurrency=1)
        raise AssertionError("expected the run to fail")
    except RuntimeError:
        pass
    assert len((tmp_path / "a@1.part.jsonl").read_text().splitlines()) == 6     # every answer kept as it landed
    with open(tmp_path / "b@1.part.jsonl", "a") as f:
        f.write('{"id": "b9", "k"')                             # a torn last line
    sent.clear()
    run_stream(reqs(), lambda p, s: (sent.append(p), ("x ok", "stop", 1))[1], grade, tmp_path, concurrency=2)
    assert sorted(sent) == ["Q b2", "Q b3"]                     # only what was missing
    assert len((tmp_path / "b@1.jsonl").read_text().splitlines()) == 4 and (tmp_path / "a@1.jsonl").exists()


def test_an_answer_the_grader_cannot_read_is_wrong_not_a_failed_run(tmp_path):
    def picky(r, text, finish):
        if r["id"] == "b1":
            raise TimeoutError("regular-expression match exceeded its 5.0s time limit")
        return grade(r, text, finish)

    run_stream(reqs(), lambda p, s: (p + " ok", "length", 1), picky, tmp_path, concurrency=2)
    rows = {r["id"]: r for r in map(json.loads, (tmp_path / "b@1.jsonl").read_text().splitlines())}
    assert rows["b1"]["correct"] is False and rows["b1"]["kind"] == "grade_error" and rows["b1"]["cut"]
    assert all(rows[i]["correct"] for i in ("b0", "b2", "b3"))


def test_a_finished_benchmark_is_not_run_again(tmp_path):
    run_stream(reqs(), lambda p, s: ("x ok", "stop", 1), grade, tmp_path, concurrency=2)
    before = (tmp_path / "a@1.jsonl").read_text()
    sent = []
    run_stream(reqs(), lambda p, s: (sent.append(p), ("x no", "stop", 1))[1], grade, tmp_path, concurrency=2)
    assert sent == [] and (tmp_path / "a@1.jsonl").read_text() == before
    assert not list(tmp_path.glob("*.part.jsonl"))


def test_grading_runs_in_the_main_thread(tmp_path):
    seen = set()

    def main_only(r, text, finish):     # as math_verify's signal.alarm: fails outside the main thread
        seen.add(threading.current_thread() is threading.main_thread())
        return {"correct": text.endswith("ok") and threading.current_thread() is threading.main_thread()}

    run_stream(reqs(), lambda p, s: ("x ok", "stop", 1), main_only, tmp_path, concurrency=3)
    assert seen == {True}
    rows = [json.loads(x) for x in (tmp_path / "a@1.jsonl").read_text().splitlines()]
    assert all(r["correct"] for r in rows) and all(r["text"] == "x ok" for r in rows)


def test_a_grader_that_cannot_run_stops_the_run_instead_of_scoring_wrong(tmp_path):
    import pytest

    class GraderError(RuntimeError):
        pass

    def broken(r, text, finish):
        if r["id"] == "a1":
            raise GraderError("nltk data missing")
        if r["id"] == "a2":
            raise ValueError("an answer the checker cannot read")
        return {"correct": True}
    with pytest.raises(GraderError):
        run_stream(reqs(), lambda p, s: (p + " ok", "stop", 3), broken, tmp_path, concurrency=1, fatal=(GraderError,))
    rows = [json.loads(x) for x in (tmp_path / "a@1.part.jsonl").read_text().splitlines()]
    assert {r["id"] for r in rows} == {"a0"}                    # nothing scored for the item the grader failed on
