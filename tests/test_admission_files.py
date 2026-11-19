"""Admission's file check, against the pinned toolkit's own rules.json (no torch needed): an upload holds the upload
protocol's files (and its parent's), whatever the submission section allows a miner's directory to hold."""
import json
from pathlib import Path

from axonry.evaluation.admit_remote import static_checks

RULES = json.loads((Path(__file__).resolve().parents[1] / "WeightSpaceProgramming" / "rules.json").read_text())


def violations(tmp_path, files):
    for f in files:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text("x")
    v, _ = static_checks(tmp_path, {"ops": [{"op": RULES["edit"]["ops"][0]}]}, RULES)
    return [x for x in v if "upload file" in x]


def test_an_upload_and_its_parent_pass_the_file_check(tmp_path):
    assert violations(tmp_path, ["edit.json", "edit.npz", "REPORT.md",
                                 "parent/edit.json", "parent/edit.npz", "parent/REPORT.md"]) == []


def test_anything_else_in_an_upload_is_refused(tmp_path):
    assert violations(tmp_path, ["edit.json", "REPORT.md", "weights.safetensors"]) == \
        ["weights.safetensors: not an upload file (REPORT.md, edit.json, edit.npz)"]
