"""Where the benchmark items come from: the loaders that build each benchmark (`task` "bench:<name>" in the
registry) from its public source. They live here, with the organizer, and not in the competition toolkit: the toolkit
holds the practice sets only, and its `bench:<name>` reads the items the organizer supplies
($WSP_WORK/data/bench_<name>.jsonl, written by evaluation/benchdata.py). This module is left out of the setup bundle
(pool/machine.py), so a run's machine never holds it.

Needs the competition toolkit on the path (practice data for the synthetic long-context sets, the user-guess
transform) and `datasets`. Items are exactly what the toolkit built before the move, so frozen manifests still hold.
"""
import random
import re

from wsp.data import _practice, _suggest, fingerprint


def _hf(name, *args, **kw):
    from datasets import load_dataset  # noqa: PLC0415
    return load_dataset(name, *args, **kw)


def _math500():
    return [{"id": "math500_" + r["unique_id"].replace("/", "_").replace(".json", ""), "problem": r["problem"],
             "answer": r["answer"], "subject": r["subject"], "level": r["level"]}
            for r in _hf("HuggingFaceH4/MATH-500", split="test")]


def _gsm8k():
    return [{"id": f"gsm8k_test_{i}", "problem": r["question"], "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
            for i, r in enumerate(_hf("openai/gsm8k", "main", split="test"))]


def _qa(repo, split, prefix, q="question", a="answer"):
    return lambda: [{"id": f"{prefix}_{i}", "problem": r[q], "answer": str(r[a])} for i, r in enumerate(_hf(repo, split=split))]


BBH_TASKS = tuple(sorted(("boolean_expressions", "date_understanding", "formal_fallacies", "geometric_shapes",
             "hyperbaton", "logical_deduction_three_objects", "logical_deduction_five_objects",
             "logical_deduction_seven_objects", "multistep_arithmetic_two", "navigate", "object_counting",
             "penguins_in_a_table", "reasoning_about_colored_objects", "temporal_sequences",
             "tracking_shuffled_objects_three_objects", "tracking_shuffled_objects_five_objects",
             "tracking_shuffled_objects_seven_objects", "web_of_lies", "word_sorting")))   # the toolkit's order


def _bbh(per_task=26, seed=0, revision="982bb89fd79532a8ac676a61fc42eb1aeec63f99"):
    """BIG-Bench Hard: the tasks the toolkit generates practice for (wsp.puzzles; no common-sense or taste tasks, no
    dyck_languages, whose brace answers cannot sit in \\boxed{}), per_task test items each in a seeded order, a
    repeated question once."""
    rows = []
    for task in BBH_TASKS:
        d = _hf("lukaemon/bbh", task, split="test", revision=revision)
        seen, idx = set(), []
        for i in random.Random(f"{seed}:{task}").sample(range(len(d)), len(d)):
            f = fingerprint(d[i]["input"])
            if f not in seen:
                seen.add(f)
                idx.append(i)
            if len(idx) == per_task:
                break
        rows += [{"id": f"bbh_{task}_{i}", "source": "bbh", "task": "exact", "subject": task,
                  "problem": d[i]["input"], "answer": d[i]["target"]} for i in sorted(idx)]
    return rows


def _bbh_all(revision="982bb89fd79532a8ac676a61fc42eb1aeec63f99"):
    """Every BIG-Bench Hard test question, all its tasks: not a benchmark, the wider set the toolkit's practice is kept
    apart from (fingerprints)."""
    from datasets import get_dataset_config_names  # noqa: PLC0415
    return [{"id": f"bbh_{task}_{i}", "source": "bbh", "task": "exact", "subject": task, "problem": r["input"],
             "answer": r["target"]}
            for task in sorted(get_dataset_config_names("lukaemon/bbh", revision=revision))
            for i, r in enumerate(_hf("lukaemon/bbh", task, split="test", revision=revision))]


def _arc(n=500, seed=0):  # ARC-Challenge test: another multiple-choice source (grade-school science)
    d = _hf("allenai/ai2_arc", "ARC-Challenge", split="test")
    out = []
    for i in random.Random(seed).sample(range(len(d)), n):
        ch, labels = d[i]["choices"], d[i]["choices"]["label"]
        out.append({"id": f"arc_{d[i]['id']}", "problem": d[i]["question"],
                    "answer": chr(65 + labels.index(d[i]["answerKey"])),
                    "options": "\n".join(f"{chr(65 + j)}. {t}" for j, t in enumerate(ch["text"]))})
    return out


def gsmplus_rows(part, kind="distract", seed=0):
    """GSM-Plus "distraction insertion": a GSM8K test problem with an irrelevant sentence (with numbers) added;
    kind "seed" is the same original problem under the same id, so the two pair. Parts of the 1,319 problems:
    "test" 500 (the benchmark), "dev" 300 (choosing an edit), "fit" the rest (fitting directions)."""
    d = [r for r in _hf("qintongli/GSM-Plus", split="test") if r["perturbation_type"] == "distraction insertion"]
    order = list(range(len(d)))
    test = set(random.Random(seed).sample(order, 500))
    rest = [i for i in order if i not in test]
    random.Random(seed + 1).shuffle(rest)
    idx = {"test": sorted(test), "dev": sorted(rest[:300]), "fit": sorted(rest[300:])}[part]
    out = []
    for i in idx:
        r = d[i]
        q, a = (r["question"], r["answer"]) if kind == "distract" else (r["seed_question"], r["seed_answer"])
        out.append({"id": f"gsmplus_{i}", "problem": q, "answer": str(a).replace(",", ""),
                    "seed_problem": r["seed_question"]})
    return out


def _mmlupro(n=500, seed=0):
    d = _hf("TIGER-Lab/MMLU-Pro", split="test")
    return [{"id": f"mmlupro_{d[i]['question_id']}", "problem": d[i]["question"], "answer": d[i]["answer"],
             "options": "\n".join(f"{chr(65 + j)}. {o}" for j, o in enumerate(d[i]["options"])),
             "category": d[i]["category"]} for i in random.Random(seed).sample(range(len(d)), n)]


MGSM_LANGS = ("en", "de", "fr", "es", "ru", "zh", "ja", "th", "sw", "bn", "te")


def _mgsm(per_lang=50, seed=0):  # the same problems in every language, so language gaps are paired
    idx = sorted(random.Random(seed).sample(range(250), per_lang))
    rows = []
    for lang in MGSM_LANGS:
        d = _hf("juletxara/mgsm", lang, split="test")
        rows += [{"id": f"mgsm_{lang}_{i}", "problem": d[i]["question"], "answer": str(d[i]["answer_number"]),
                  "subject": lang} for i in idx]
    return rows


def _niah(lengths=(4000, 8000, 16000, 30000), per_length=50, needles=24, seed=0):
    """Multi-key needle in a haystack (RULER style): many "magic number for <key>" needles in long filler text
    (GSM8K train questions), one key asked. Lengths are approximate tokens (4 characters each)."""
    rng = random.Random(seed)
    filler = [r["problem"].replace("\n", " ") for r in _practice("gsm8k_train")]
    words = sorted({w.lower() for t in filler[:3000] for w in re.findall(r"[A-Za-z]{6,}", t)})
    rows = []
    for n_tok in lengths:
        for i in range(per_length):
            text, start = [], rng.randrange(len(filler))
            while sum(map(len, text)) < n_tok * 4:
                text.append(filler[(start + len(text)) % len(filler)])
            keys = rng.sample(words, needles)
            vals = [str(rng.randrange(1_000_000, 10_000_000)) for _ in keys]
            for k, v in zip(keys, vals):
                text.insert(rng.randrange(len(text) + 1), f"One of the special magic numbers for {k} is: {v}.")
            j = rng.randrange(needles)
            rows.append({"id": f"niah_{n_tok}_{i}", "subject": str(n_tok), "answer": vals[j],
                         "problem": "Some special magic numbers are hidden within the following text. Make sure to "
                                    "memorize them.\n\n" + " ".join(text) +
                                    f"\n\nWhat is the special magic number for {keys[j]} mentioned in the text? "
                                    "Give it within \\boxed{}."})
    return rows


def _haystack(rng, n_tok):
    filler = [r["problem"].replace("\n", " ") for r in _practice("gsm8k_train")]
    text, start = [], rng.randrange(len(filler))
    while sum(map(len, text)) < n_tok * 4:
        text.append(filler[(start + len(text)) % len(filler)])
    return text


def _longctx(kind, lengths=(16000, 30000), per_length=50, seed=3):
    """RULER-style variants, answers are sorted word lists: "mv" all values of one key among other keys,
    "vt" all variables bound (directly or through a 4-hop chain) to one value."""
    rng, rows = random.Random(seed), []
    names = ["".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5)) for _ in range(4000)]
    for n_tok in lengths:
        for i in range(per_length):
            text = _haystack(rng, n_tok)
            if kind == "mv":
                keys = rng.sample(names, 6)
                vals = {k: [str(rng.randrange(1_000_000, 10_000_000)) for _ in range(4)] for k in keys}
                for k in keys:
                    for v in vals[k]:
                        text.insert(rng.randrange(len(text) + 1), f"One of the special magic numbers for {k} is: {v}.")
                k = keys[0]
                q = (f"What are all the special magic numbers for {k} mentioned in the text? List them in ascending "
                     "order, separated by spaces, within \\boxed{}.")
                ans = " ".join(sorted(vals[k], key=int))
            else:
                chains = []
                for _ in range(3):
                    vs, val = rng.sample(names, 5), str(rng.randrange(10_000, 100_000))
                    chains.append((vs, val, [f"VAR {vs[0]} = {val}."] + [f"VAR {vs[j]} = VAR {vs[j - 1]}." for j in range(1, 5)]))
                for _, _, stmts in chains:  # keep each chain in order, spread over the text
                    pos = sorted(rng.randrange(len(text) + 1) for _ in stmts)
                    for off, (p_, st) in enumerate(zip(pos, stmts)):
                        text.insert(p_ + off, st)
                vs, val, _ = chains[0]
                q = (f"Find all variables that are assigned the value {val}, directly or through other variables. List "
                     "them in alphabetical order, separated by spaces, within \\boxed{}.")
                ans = " ".join(sorted(vs))
            rows.append({"id": f"{kind}_{n_tok}_{i}", "subject": str(n_tok), "answer": ans,
                         "problem": "Memorize the information hidden in the following text.\n\n" + " ".join(text) +
                                    "\n\n" + q})
    return rows


def _ifeval():
    return [{"id": f"ifeval_{r['key']}", "problem": r["prompt"], "answer": None,
             "instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"]}
            for r in _hf("google/IFEval", split="train")]


def _github(repo, commit, path, sha256):
    """A file of a GitHub repository at a pinned commit, checked against its sha256."""
    import hashlib  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415
    with urllib.request.urlopen(f"https://raw.githubusercontent.com/{repo}/{commit}/{path}", timeout=120) as r:
        data = r.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"{repo}@{commit[:12]}/{path}: sha256 {hashlib.sha256(data).hexdigest()[:16]}, not the pinned")
    return data


def _ifbench():
    """IFBench (AllenAI, 2025): its 300 test prompts, graded by its own checkers (the toolkit's task "ifbench")."""
    import json  # noqa: PLC0415
    raw = _github("allenai/IFBench", "488918665d53f867fa908a293769ec6e71e74aec", "data/IFBench_test.jsonl",
                  "d2ada7da94a38cfe406351614c4e686846ed2da6d1b339db95fa5ead19554a4a")
    return [{"id": f"ifbench_{r['key']}", "task": "ifbench", "problem": r["prompt"], "answer": None,
             "instruction_id_list": r["instruction_id_list"],
             "kwargs": [{k: v for k, v in (d or {}).items() if v is not None} for d in r["kwargs"]]}
            for r in map(json.loads, raw.decode().splitlines()) if r]


def _instruct600(n_ifeval=300, seed=0):
    """Instruction following: n_ifeval IFEval prompts (a seeded sample, in IFEval's order) and every IFBench one."""
    ife = _ifeval()
    return [{**ife[i], "task": "ifeval"} for i in sorted(random.Random(seed).sample(range(len(ife)), n_ifeval))] + _ifbench()


def _gpqa(subset="diamond", seed=0):
    """GPQA (Rein et al. 2023) from its authors' repository (dataset.zip, password as its README gives it): the four
    answers of each question in a seeded order, the letter of the correct one the answer."""
    import csv  # noqa: PLC0415
    import io  # noqa: PLC0415
    import zipfile  # noqa: PLC0415
    z = zipfile.ZipFile(io.BytesIO(_github("idavidrein/gpqa", "56686c06f5e19865c153de0fdb11be3890014df7", "dataset.zip",
                                           "461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc")))
    text = z.read(f"dataset/gpqa_{subset}.csv", pwd=b"deserted-untie-orchid").decode()
    rng, out = random.Random(seed), []
    for r in csv.DictReader(io.StringIO(text)):
        answers = [r["Correct Answer"]] + [r[f"Incorrect Answer {k}"] for k in (1, 2, 3)]
        order = rng.sample(range(4), 4)
        out.append({"id": f"gpqa_{r['Record ID']}", "problem": r["Question"].strip(),
                    "answer": chr(65 + order.index(0)),
                    "options": "\n".join(f"{chr(65 + j)}. {answers[k].strip()}" for j, k in enumerate(order)),
                    "subject": r["High-level domain"]})
    return out


# name -> (loader, task, number of items kept (None = all))
BENCHMARKS = {
    "math500": (_math500, "math", None),
    "gsm8k": (_gsm8k, "math", None),
    "gsm8k500": (_gsm8k, "math", 500),
    "amc23": (_qa("math-ai/amc23", "test", "amc23"), "math", None),
    "aime24": (_qa("HuggingFaceH4/aime_2024", "train", "aime24", q="problem"), "math", None),
    "aime25": (_qa("MathArena/aime_2025", "train", "aime25", q="problem"), "math", None),
    "mmlupro500": (_mmlupro, "choice", None),
    "ifeval": (_ifeval, "ifeval", None),
    "ifbench": (_ifbench, "ifbench", None),
    "instruct600": (_instruct600, None, None),        # each item's own task: "ifeval" or "ifbench"
    "gpqa": (_gpqa, "choice", None),
    "gpqa_all": (lambda: _gpqa("extended"), "choice", None),  # fingerprints only (main and diamond are in it)
    "bbh": (_bbh, "exact", None),
    "bbh_all": (_bbh_all, "exact", None),  # fingerprints only
    "mgsm": (_mgsm, "math", None),
    "niah": (_niah, "exact", None),
    "niah30k_sel": (lambda: _niah(lengths=(30000,), per_length=100, seed=1), "exact", None),  # for searches
    "mv": (lambda: _longctx("mv"), "exact", None),
    "vt": (lambda: _longctx("vt"), "exact", None),
    "mv30k_sel": (lambda: _longctx("mv", lengths=(30000,), per_length=100, seed=4), "exact", None),  # for searches
    "niah30k_conf": (lambda: _niah(lengths=(30000,), per_length=100, seed=2), "exact", None),  # their confirmation
    "mmlupro500_sug": (lambda: _suggest(rows("mmlupro500")), "choice", None),  # sycophancy: a wrong user guess
    "mmlupro500_sugok": (lambda: _suggest(rows("mmlupro500"), right=True), "choice", None),  # ... a right one
    "mmlupro500_sugvar": (lambda: _suggest(rows("mmlupro500"), variants=True), "choice", None),  # other phrasings
    "arc500": (_arc, "choice", None),
    "gsmplus500_seed": (lambda: gsmplus_rows("test", "seed"), "math", None),  # the originals of gsmplus500_distract
    "gsmplus500_distract": (lambda: gsmplus_rows("test"), "math", None),  # + an irrelevant sentence with numbers
    "gsmplus_dev_seed": (lambda: gsmplus_rows("dev", "seed"), "math", None),
    "gsmplus_dev_distract": (lambda: gsmplus_rows("dev"), "math", None),
    "arc500_sug": (lambda: _suggest(rows("arc500")), "choice", None),
    "arc500_sugok": (lambda: _suggest(rows("arc500"), right=True), "choice", None),
    "gsm8k500_sug": (lambda: _suggest(rows("gsm8k500")), "math", None),  # transfer to another domain
}


def rows(name):
    """The items of benchmark <name>, as the toolkit reads them (each with its "source" and "task")."""
    loader, task, limit = BENCHMARKS[name]
    out = loader()[:limit]
    for r in out:
        r.setdefault("source", name)
        r.setdefault("task", task)
    return out


FINGERPRINT_BENCHES = ("math500", "gsm8k", "amc23", "aime24", "aime25", "mmlupro500", "ifeval", "bbh_all", "ifbench",
                       "gpqa_all")


def fingerprints(names=FINGERPRINT_BENCHES):
    """The question fingerprints of these benchmarks: the toolkit's data/bench_fingerprints.txt.gz (one per line),
    against which its practice sets are kept disjoint."""
    return sorted({fingerprint(r["problem"]) for b in names for r in rows(b)})
