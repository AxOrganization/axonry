import json

import pytest


def test_axonry_check_refuses_a_parent_in_another_form(tmp_path):
    """Admission accepts a start's sid or the base's full name only: the check says so before the run ends."""
    from axonry import sandbox_tools
    task = {"base": "b", "starts": [{"sid": "0009538688-0019", "role": "champion"}], "base_name": "Org/B@1"}
    (tmp_path / "task.json").write_text(json.dumps(task))
    up = tmp_path / "upload"
    up.mkdir()
    (up / "edit.json").write_text(json.dumps({"base": "b", "parent": "0009538688-0019/edit.json", "ops": []}))
    (tmp_path / "0009538688-0019" / "edit").mkdir(parents=True)          # it is there, named another way
    with pytest.raises(SystemExit, match='"0009538688-0019" or "Org/B@1"'):
        sandbox_tools.main(["check", str(up), "--task", str(tmp_path / "task.json"), "--base-dir", "unused"])


@pytest.mark.parametrize("bad", ["missing", "upload/edit.json"])
def test_axonry_check_says_which_directory_to_give(tmp_path, bad):
    from axonry import sandbox_tools
    (tmp_path / "task.json").write_text(json.dumps({"base": "b"}))
    up = tmp_path / "upload"
    up.mkdir()
    (up / "edit.json").write_text(json.dumps({"base": "b", "ops": []}))
    with pytest.raises(SystemExit, match="edit/ directory"):
        sandbox_tools.main(["check", str(tmp_path / bad), "--task", str(tmp_path / "task.json"), "--base-dir", "x"])


def test_the_parent_is_a_start_of_the_bundle_by_its_id_or_the_base_by_its_full_name():
    from axonry.sandbox_tools import start_of
    task = {"starts": [{"sid": "P1", "role": "parent"}, {"sid": "C1", "role": "champion"}], "base_name": "Org/B@1"}
    assert start_of(task, "C1") == "C1" and start_of(task, "Org/B@1") is None
    for bad in (None, "C1/edit.json", "parent/edit.json", "X9", "Org/B"):   # never left out, never another
        with pytest.raises(SystemExit, match='"P1" or "C1" or "Org/B@1"'):
            start_of(task, bad)


def test_an_archived_starts_lineage_note_is_not_an_upload_file_but_may_sit_in_a_placed_start(tmp_path):
    from axonry.evaluation.admit_remote import static_checks
    rules = json.load(open(__file__.rsplit("/", 2)[0] + "/WeightSpaceProgramming/rules.json"))
    for f in ("edit.json", "REPORT.md", "S1/edit/edit.json", "S1/edit/REPORT.md", "S1/edit/FLAT.json"):
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text("{}")
    assert not [v for v in static_checks(tmp_path, {"ops": []}, rules)[0] if "not an upload file" in v]
    (tmp_path / "FLAT.json").write_text("{}")                     # in the upload itself: not an upload file
    assert [v for v in static_checks(tmp_path, {"ops": []}, rules)[0] if "not an upload file" in v] == [
        "FLAT.json: not an upload file (REPORT.md, edit.json, edit.npz)"]


def test_axonry_check_refuses_an_upload_over_the_size_limit(tmp_path):
    from axonry import sandbox_tools
    from axonry.evaluation.staging import MAX_TOTAL_BYTES
    (tmp_path / "task.json").write_text(json.dumps({"base": "b", "starts": [], "base_name": "Org/B@1"}))
    up = tmp_path / "upload"
    up.mkdir()
    (up / "edit.json").write_text(json.dumps({"base": "b", "parent": "Org/B@1", "ops": [{"values": [0.5] * 10}]}))
    (up / "edit.npz").write_bytes(b"\0" * MAX_TOTAL_BYTES)
    with pytest.raises(SystemExit, match="over the 8,388,608-byte limit"):
        sandbox_tools.main(["check", str(up), "--task", str(tmp_path / "task.json"), "--base-dir", "unused"])
