"""Admission against the real competition toolkit (wsp.edits.build / spec_size) on a tiny synthetic model.
Skipped when torch/safetensors or the competition repo are not available."""
import json
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
COMP = Path(os.environ.get("AXONRY_COMPETITION_REPO", Path(__file__).resolve().parents[1] / "WeightSpaceProgramming"))
if not (COMP / "wsp" / "edits.py").exists():
    pytest.skip("competition repo not available", allow_module_level=True)
sys.path.insert(0, str(COMP))

from axonry.evaluation.admit_remote import admit  # noqa: E402

BASE_ID = "qwen35-0.8b-think"


@pytest.fixture()
def base(tmp_path):
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(0)
    t = {f"model.layers.{i}.mlp.down_proj.weight": torch.randn(8, 16, generator=g).to(torch.bfloat16) for i in range(4)}
    t["model.embed_tokens.weight"] = torch.randn(32, 8, generator=g).to(torch.bfloat16)
    t["model.layers.0.input_layernorm.weight"] = torch.ones(8, dtype=torch.bfloat16)
    d = tmp_path / "base"
    d.mkdir()
    save_file(t, str(d / "model.safetensors"), metadata={"format": "pt"})
    (d / "config.json").write_text("{}")
    return d


def spec_dir(tmp_path, name, ops, parent=None):
    d = tmp_path / name
    d.mkdir()
    spec = {"name": "edit", "base": BASE_ID, "ops": ops, **({"parent": parent} if parent else {})}
    (d / "edit.json").write_text(json.dumps(spec))
    (d / "REPORT.md").write_text("report")
    return d


def ablate(layer, alpha, cols=None):
    u = [1.0] + [0.0] * 7
    return {"op": "dir_ablate", "tensor": f"model.layers.{layer}.mlp.down_proj.weight", "u": u, "alpha": alpha,
            "side": "out", **({"cols": cols} if cols else {})}


def test_small_edit_passes(tmp_path, base):
    r = admit(spec_dir(tmp_path, "ok", [ablate(1, 0.1, cols=[0, 1])]), base, BASE_ID)
    assert r["passed"], r["violations"]
    assert r["model_digest"] and r["info"]["tensors"] == 1


def test_large_change_rejected_with_reason(tmp_path, base):
    r = admit(spec_dir(tmp_path, "big", [ablate(1, 1.0), ablate(2, 1.0), ablate(3, 1.0)]), base, BASE_ID)
    assert not r["passed"]
    assert any("global relative change" in v for v in r["violations"])


def test_forbidden_tensor_and_base_mismatch(tmp_path, base):
    bad = {"op": "scale_rows", "tensor": "model.embed_tokens.weight", "rows": [0], "factor": 2.0}
    r = admit(spec_dir(tmp_path, "emb", [bad]), base, BASE_ID)
    assert not r["passed"] and any("not inside a main decoder layer" in v for v in r["violations"])
    r = admit(spec_dir(tmp_path, "other", [ablate(1, 0.1)]), base, "another-base")
    assert not r["passed"] and "base" in r["violations"][0]


def test_missing_report(tmp_path, base):
    d = spec_dir(tmp_path, "norep", [ablate(1, 0.1)])
    (d / "REPORT.md").unlink()
    r = admit(d, base, BASE_ID)
    assert not r["passed"] and any("REPORT.md" in v for v in r["violations"])


def test_stacked_edit_limits_only_new_layer(tmp_path, base):
    parent = spec_dir(tmp_path, "parent", [ablate(1, 0.3, cols=list(range(8)))])
    child = spec_dir(tmp_path, "child", [ablate(2, 0.1, cols=[0, 1])], parent="parent/edit.json")
    import shutil
    shutil.copytree(parent, child / "parent")
    r = admit(child, base, BASE_ID)
    assert r["passed"], r["violations"]
    assert r["info"]["tensors"] == 1                       # only the child's own ops are counted
    assert len(r["info"]["chain"]) == 2


def test_axonry_check_builds_a_stacked_upload_on_the_bundles_parent(tmp_path, base):
    from axonry import sandbox_tools
    parent = spec_dir(tmp_path, "bundle_parent", [ablate(1, 0.1, cols=[0, 1])])
    sid = "0000000000-0000"
    (tmp_path / "task.json").write_text(json.dumps({"base": BASE_ID, "starts": [{"sid": sid, "role": "parent"}],
                                                    "base_name": "unused@0"}))
    (tmp_path / sid).mkdir()
    parent.rename(tmp_path / sid / "edit")                  # as the bundle holds it
    (tmp_path / sid / "edit" / "FLAT.json").write_text("{}")  # an archived start keeps its lineage note
    up = spec_dir(tmp_path, "upload", [ablate(2, 0.1, cols=[0, 1])], parent=sid)
    args = ["check", str(up), "--task", str(tmp_path / "task.json"), "--base-dir", str(base)]
    assert sandbox_tools.main(args) == 0
    assert not (up / sid).exists()                          # the upload itself is left as it is

