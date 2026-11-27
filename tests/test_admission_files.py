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


def test_the_payload_limit_grows_with_the_base_width_from_the_registry_hidden_size(tmp_path):
    from axonry import registry
    from axonry.agent.brief import limits_from
    from axonry.evaluation.admit_remote import hidden_size, payload_limit
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"hidden_size": 2048}}))
    assert payload_limit(RULES, hidden_size(tmp_path)) == 8 << 20
    assert [limits_from(RULES, registry.load_base(b)["hidden_size"])["max_payload_bytes"]
            for b in ("qwen35-0.8b-think", "qwen35-2b-think")] == [4 << 20, 8 << 20]
