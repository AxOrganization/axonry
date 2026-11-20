"""Commands available to the agent on the run's machine (installed by pool/setup.sh with this package):

  axonry-time                      time left until the run's deadline (/etc/axonry-deadline, written when the
                                   machine is handed to the run)
  axonry-done                      the agent is finished (a complete result is uploaded): the run ends early
  axonry-check OUT_DIR --task task.json   the organizer's admission check on OUT_DIR (edit.json, edit.npz, REPORT.md);
                                   a stacked edit is checked on the parent chain (--parent, default parent/edit
                                   beside task.json)
Uploading is the competition toolkit's job: python -m wsp.upload <case> <edit> --task task.json (schema: the protocol's upload.schema).
"""
import argparse
import json
import sys
import time
from pathlib import Path



def time_left(_a):
    p = Path("/etc/axonry-deadline")
    dl = int(p.read_text().strip()) if p.exists() else 0
    left = dl - int(time.time())
    print(f"{max(left, 0) // 3600} h {max(left, 0) % 3600 // 60} min left (deadline "
          f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(dl))})" if dl else "no deadline set on this machine")


DONE = Path("/workspace/work/DONE")      # read by the organizer (pool.machine.sample)


def done(_a):
    DONE.write_text(f"{int(time.time())}\n")
    print("noted: the run ends within a minute, with the last complete upload (if any)")


def check(a):
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    from .evaluation.admit_remote import admit  # noqa: PLC0415

    task_path = Path(a.task)
    if not task_path.exists():
        raise SystemExit(f"{task_path} not found: copy the bundle's task.json here or pass --task")
    task, out = json.loads(task_path.read_text()), Path(a.out_dir)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        ref = json.loads((out / "edit.json").read_text()).get("parent") if (out / "edit.json").exists() else None
        name = Path(ref).parent.as_posix() if ref else None      # "parent" or "champion": the start it stacks on
        if name and name not in ("parent", "champion"):        # what admission accepts, nothing else
            raise SystemExit(f'edit.json parent must be "parent/edit.json" or "champion/edit.json" (the bundle\'s '
                             f'parent/edit/ or champion/edit/, kept under that name), not {ref!r}')
        if name and not (out / name).exists():   # uploads only reference the published start
            parent = Path(a.parent) if a.parent else task_path.parent / name / "edit"
            if not (parent / "edit.json").exists():
                raise SystemExit(f"the edit it stacks on is not at {parent}: copy the bundle's {name}/edit/ there or "
                                 "pass --parent")
            out = shutil.copytree(out, Path(tmp) / "out")
            shutil.copytree(parent, out / name)
        res = admit(str(out), a.base_dir, task["base"])
    print(json.dumps(res, indent=1))
    return 0 if res["passed"] else 1


def main(argv=None):
    prog = Path(sys.argv[0]).name
    ap = argparse.ArgumentParser(prog=prog)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("time")
    sub.add_parser("done")
    c = sub.add_parser("check")
    c.add_argument("out_dir")
    c.add_argument("--task", default="task.json")
    c.add_argument("--parent", help="the edit it stacks on (default: parent/edit or champion/edit beside task.json, "
                                    "as its edit.json names)")
    c.add_argument("--base-dir", default="/workspace/base")  # the verified base, read-only
    argv = list(sys.argv[1:] if argv is None else argv)
    alias = {"axonry-time": "time", "axonry-done": "done", "axonry-check": "check"}.get(prog)
    if alias:
        argv = [alias, *argv]
    a = ap.parse_args(argv)
    return {"time": time_left, "done": done, "check": check}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
