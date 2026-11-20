import json

import pytest


def test_axonry_check_refuses_a_parent_under_another_name(tmp_path):
    """Admission accepts parent/edit.json or champion/edit.json only: the check says so before the run ends."""
    from axonry import sandbox_tools
    (tmp_path / "task.json").write_text(json.dumps({"base": "b", "parent": {"sid": "0009538688-0019"}}))
    up = tmp_path / "upload"
    up.mkdir()
    (up / "edit.json").write_text(json.dumps({"base": "b", "parent": "0009538688-0019/edit.json", "ops": []}))
    (tmp_path / "0009538688-0019" / "edit").mkdir(parents=True)          # it is there, under another name
    with pytest.raises(SystemExit, match="parent/edit.json"):
        sandbox_tools.main(["check", str(up), "--task", str(tmp_path / "task.json"), "--base-dir", "unused"])
