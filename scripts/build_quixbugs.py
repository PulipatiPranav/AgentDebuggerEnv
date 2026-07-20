"""Adapt QuixBugs into a secondary, out-of-distribution eval set (Publication-Strategy §1.6).

Why this exists
----------------
Every bug in the packaged dataset (v1 and v2) was authored by this project — by
hand-templating LeetCode problems, or by AST-mutating MBPP solutions. A held-out
*split* proves a model was not trained on these exact bugs; it does not prove the
model has not simply learned this project's own bug idioms (a particular mutation
operator's signature, this codebase's error-message phrasing, etc.). QuixBugs is
a well-known program-repair benchmark (Lin et al., "QuixBugs: A Multi-Lingual
Program Repair Benchmark", ISSTA 2017) that this project did not author and did
not curate: each program has a single, naturally-occurring one-line defect found
in the wild (the Quixey Challenge), independent of anything here. Solving it is
the closest thing to evidence the model learned to *debug*, not to pattern-match
this project's dataset.

What this does
---------------
1. Obtains a QuixBugs checkout — either ``--quixbugs-dir`` (a local clone) or, by
   default, a shallow ``git clone`` into a scratch directory.
2. Keeps only the programs QuixBugs itself supplies as **single-function, scalar
   or list I/O** (a JSON-representable ``json_testcases/<name>.json`` file): the
   graph/linked-list programs (``breadth_first_search``, ``reverse_linked_list``,
   ...) take a custom ``Node`` object as an argument, which is not something this
   project's ``TestCase`` (JSONL, JSON-stable ``input``/``expected_output``) can
   represent, and two programs (``flatten``, ``kheapsort``) are generators, whose
   return value ``==`` a list is always ``False`` even when correct. All three
   exclusions are structural, not a difficulty judgement — see ``_EXCLUDED`` below
   for the full list and reasons.
3. Extracts *only* the function body (QuixBugs appends a docstring with the
   problem statement and doctest examples after the code; that is dropped, so
   ``buggy_code`` looks like every other bug in this dataset: a function and
   nothing else).
4. Adapts each surviving program's ``json_testcases`` lines into ``TestCase``
   records and validates every one through the same
   :func:`agentdebugger.dataset.validate.validate_bug` this project's own dataset
   is checked with (reference passes every case, buggy fails at least one).

Output: ``data/quixbugs/bugs.jsonl`` plus ``data/quixbugs/DATACARD.md``.

Usage:
    python scripts/build_quixbugs.py                       # clones QuixBugs itself
    python scripts/build_quixbugs.py --quixbugs-dir ~/QuixBugs --out data/quixbugs
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from agentdebugger.dataset.models import Bug
from agentdebugger.dataset.validate import _VALIDATION_POLICY, validate_bug
from agentdebugger.sandbox import run_test_cases

QUIXBUGS_REPO = "https://github.com/jkoppel/QuixBugs.git"

#: Programs QuixBugs ships that this project cannot adapt, and why. Excluding a
#: program never depends on how hard its bug is — only on whether its I/O and
#: return value can be represented and compared the way this project's sandbox
#: harness does (``function(*input) == expected_output``, a JSON-stable value).
_EXCLUDED: dict[str, str] = {
    # These take a `Node`/linked-list/graph argument. `TestCase.input` must be a
    # JSON-stable value (it round-trips through JSONL), and a `Node` is not one.
    "breadth_first_search": "argument is a graph Node, not JSON-representable",
    "depth_first_search": "argument is a graph Node, not JSON-representable",
    "detect_cycle": "argument is a linked-list Node, not JSON-representable",
    "minimum_spanning_tree": "argument is a graph of Node/Edge, not JSON-representable",
    "reverse_linked_list": "argument is a linked-list Node, not JSON-representable",
    "shortest_path_length": "argument is a graph Node, not JSON-representable",
    "shortest_path_lengths": "argument is a graph Node, not JSON-representable",
    "shortest_paths": "argument is a graph Node, not JSON-representable",
    "topological_ordering": "argument is a graph Node, not JSON-representable",
    # These are generator functions (`yield`). `generator == list` is always
    # False, so even the *correct* reference would fail this project's
    # `func(*args) == expected` harness — a limitation of the harness, not a
    # property of the bug.
    "flatten": "reference implementation is a generator (`yield`); `gen == list` is always False",
    "kheapsort": "reference implementation is a generator (`yield`); `gen == list` is always False",
    # Returns a list of 2-tuples (step pairs). A JSON round-trip turns tuples
    # into lists, so `[(1, 3)] == [[1, 3]]` is False even for the correct
    # reference — the same `json_stable` rule scripts/build_dataset.py already
    # enforces on every bug's expected outputs, applied to an external source.
    "hanoi": "reference returns a list of tuples, which is not JSON-stable (tuple != list after round-trip)",
    # An approximate, epsilon-bounded algorithm (Newton's method). QuixBugs'
    # own published expected values are *one* value satisfying the epsilon
    # bound, not *the* value; this project's harness checks exact `==`, which
    # a numerically-equally-valid convergence path can legitimately miss.
    "sqrt": "epsilon-approximate algorithm; exact `==` against one specific published value is the wrong check",
}

#: A candidate program's reference implementation must finish one test case
#: within this many seconds (probed directly, not through the sandbox) to be
#: kept. QuixBugs bundles a couple of stress-sized cases (e.g. `knapsack`'s
#: multi-million capacity, `levenshtein`'s 50-character strings against an
#: unmemoised recursive reference) that no realistic training/eval sandbox
#: budget will ever complete; dropping only those *cases* — not the whole
#: bug — keeps everything else the case exercises.
_CASE_TIME_BUDGET_SECONDS = 2.0

#: A bug needs at least this many surviving cases after the timing filter to
#: still be a meaningful test (matches build_dataset.py's own minimum).
_MIN_CASES = 3


def _clone_quixbugs(into: Path) -> Path:
    print(f"Cloning {QUIXBUGS_REPO} into {into} ...", flush=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", QUIXBUGS_REPO, str(into)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return into


def _program_names(quixbugs_dir: Path) -> list[str]:
    """Every QuixBugs program name that is not excluded, sorted for determinism."""
    programs = {
        path.stem
        for path in (quixbugs_dir / "python_programs").glob("*.py")
        if not path.stem.endswith("_test") and path.stem != "node"
    }
    have_testcases = {path.stem for path in (quixbugs_dir / "json_testcases").glob("*.json")}
    usable = sorted(programs & have_testcases)
    return [name for name in usable if name not in _EXCLUDED]


def _function_source(path: Path, function_name: str) -> str:
    """The source of exactly one top-level function (plus any top-level imports
    it might need), dropping everything else.

    QuixBugs' program files are ``[imports] <the function>`` followed by a
    module-level docstring describing the problem and giving doctest examples.
    The docstring never belongs in ``buggy_code``/``original_code`` — every
    other bug record in this project is just a function, and the docstring
    would leak the input/output spec in a form no other bug gets. Top-level
    imports (most QuixBugs programs import locally, inside the function, but a
    few — e.g. ``to_base``'s ``import string`` — do not) are kept, or the
    function would ``NameError`` on a name that was always in scope in QuixBugs'
    own file.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [
        ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            segment = ast.get_source_segment(source, node)
            if segment is None:  # pragma: no cover - defensive
                raise ValueError(f"could not extract {function_name!r} from {path}")
            return "\n".join([*imports, segment]) if imports else segment
    raise ValueError(f"{path} does not define a top-level function {function_name!r}")


def _drop_pathologically_slow_cases(
    correct_code: str, function_name: str, cases: list[tuple[list[Any], Any]]
) -> list[tuple[list[Any], Any]] | None:
    """Probe the *correct* reference directly and drop cases it cannot finish.

    Runs outside the sandbox (no subprocess, no policy check) purely to time
    the reference; a case the reference itself cannot complete within budget
    will never validate anyway, through any sandbox. Returns ``None`` if fewer
    than :data:`_MIN_CASES` survive.

    Calls the reference on a **deep copy** of each case's args: several
    QuixBugs programs (``next_palindrome``, ``possible_change``'s coin list,
    ...) mutate their argument in place, and this probe must not leave that
    mutation behind in the case that later gets used for real — it would
    silently turn "input" into "input, already partially solved".
    """
    namespace: dict[str, Any] = {}
    exec(compile(correct_code, "<quixbugs-probe>", "exec"), namespace)  # trusted upstream source, dev-time only
    reference = namespace[function_name]

    def _handle_alarm(signum: int, frame: Any) -> None:
        raise TimeoutError

    previous_handler = signal.signal(signal.SIGALRM, _handle_alarm)
    kept = []
    try:
        for args, expected in cases:
            signal.alarm(int(_CASE_TIME_BUDGET_SECONDS))
            try:
                reference(*copy.deepcopy(args))
            except TimeoutError:
                continue
            except Exception:
                pass  # a case the reference errors on is still a real, fast case
            finally:
                signal.alarm(0)
            kept.append((args, expected))
    finally:
        signal.signal(signal.SIGALRM, previous_handler)

    return kept if len(kept) >= _MIN_CASES else None


def _load_test_cases(path: Path) -> list[tuple[list[Any], Any]]:
    """Parse ``json_testcases/<name>.json``: one JSON value ``[args, expected]`` per line."""
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        args, expected = json.loads(line)
        cases.append((args, expected))
    return cases


def _initial_error(function_name: str, buggy_code: str, cases: list[tuple[list[Any], Any]]) -> str:
    """A short, honest failure description for the first case the buggy code fails."""
    test_cases = [{"input": args, "expected_output": expected} for args, expected in cases]
    results = run_test_cases(buggy_code, function_name, test_cases, policy=_VALIDATION_POLICY)
    for (args, expected), ok in zip(cases, results.outcomes, strict=False):
        if not ok:
            arglist = ", ".join(repr(a) for a in args)
            return f"Test failed: {function_name}({arglist}) should return {expected!r}."
    return "Some tests are failing."


def build_bug(quixbugs_dir: Path, name: str) -> dict[str, Any] | None:
    """Adapt one QuixBugs program into a validated bug record, or ``None``."""
    buggy_path = quixbugs_dir / "python_programs" / f"{name}.py"
    correct_path = quixbugs_dir / "correct_python_programs" / f"{name}.py"
    cases_path = quixbugs_dir / "json_testcases" / f"{name}.json"

    buggy_code = _function_source(buggy_path, name)
    original_code = _function_source(correct_path, name)
    all_cases = _load_test_cases(cases_path)

    cases = _drop_pathologically_slow_cases(original_code, name, all_cases)
    if cases is None:
        print(
            f"  SKIP {name}: fewer than {_MIN_CASES} of its {len(all_cases)} cases finish "
            f"within {_CASE_TIME_BUDGET_SECONDS:g}s even for the reference implementation",
            flush=True,
        )
        return None
    if len(cases) < len(all_cases):
        print(f"  ({name}: dropped {len(all_cases) - len(cases)} pathologically slow case(s))", flush=True)

    error = _initial_error(name, buggy_code, cases)

    record = Bug.from_dict(
        {
            "id": f"quixbugs_{name}",
            # Tier 0 marks "outside the internal 1/2/3 tier scheme" — this set
            # is not loaded through `load_bugs`/`load_tier` and is never trained
            # on; it exists only as a secondary, external held-out eval.
            "difficulty": 0,
            "bug_type": "quixbugs",
            "function_name": name,
            "buggy_code": buggy_code,
            "original_code": original_code,
            "initial_error": error,
            # QuixBugs does not publish a line number for its one-line defects
            # in a machine-readable form; -1 means "unknown", which every
            # localization-scoring path already treats as "cannot be credited"
            # rather than as a wrong answer.
            "bug_location": {"function": name, "line_start": -1},
            "test_cases": [{"input": args, "expected_output": expected} for args, expected in cases],
        }
    ).as_dict()

    roundtripped = Bug.from_dict(json.loads(json.dumps(record)))
    report = validate_bug(roundtripped)
    if not report.ok:
        print(f"  SKIP {name}: {'; '.join(report.problems)}", flush=True)
        return None
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quixbugs-dir",
        help="path to an existing QuixBugs checkout (cloned if omitted)",
    )
    parser.add_argument("--out", default="data/quixbugs", help="output directory")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.quixbugs_dir:
        quixbugs_dir = Path(args.quixbugs_dir)
        scratch = None
    else:
        scratch = tempfile.TemporaryDirectory(prefix="quixbugs-")
        quixbugs_dir = _clone_quixbugs(Path(scratch.name) / "QuixBugs")

    try:
        names = _program_names(quixbugs_dir)
        total_seen = len(
            [
                p
                for p in (quixbugs_dir / "python_programs").glob("*.py")
                if not p.stem.endswith("_test") and p.stem != "node"
            ]
        )
        print(
            f"QuixBugs ships {total_seen} programs; {len(_EXCLUDED)} excluded "
            f"(graph/Node argument or generator return); {len(names)} candidates.\n",
            flush=True,
        )

        records = []
        for name in names:
            record = build_bug(quixbugs_dir, name)
            if record is not None:
                records.append(record)
                print(f"  OK   {name}", flush=True)

        path = out / "bugs.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        _write_datacard(out, len(names), len(records))
        print(f"\nWrote {len(records)}/{len(names)} validated bugs to {path}")
        if len(records) < len(names):
            print(f"({len(names) - len(records)} candidates failed validation; see SKIP lines above.)")
        return 0
    finally:
        if scratch is not None:
            scratch.cleanup()


def _write_datacard(out: Path, candidates: int, kept: int) -> None:
    exclusion_lines = "\n".join(f"  - ``{name}``: {reason}" for name, reason in sorted(_EXCLUDED.items()))
    (out / "DATACARD.md").write_text(
        f"""# QuixBugs transfer set (external, secondary eval)

**Source.** [QuixBugs](https://github.com/jkoppel/QuixBugs) (Lin, Koppel, Chen,
Foster, "QuixBugs: A Multi-Lingual Program Repair Benchmark Set Based on the
Quixey Challenge", ISSTA 2017 companion). MIT licensed; adapted here, not
redistributed verbatim — only this project's ``Bug``/``TestCase`` records are
committed, generated by ``scripts/build_quixbugs.py`` from an upstream clone.

**Why it exists.** Every bug in this project's own dataset (v1, v2) was authored
or generated by this project. QuixBugs was not: each program has one
naturally-occurring, one-line defect from the original Quixey Challenge,
independent of this codebase's mutation operators, phrasing, or idioms. A model
that solves *this* set is not simply pattern-matching something it may have
learned about how this project builds bugs (Publication-Strategy §1.6).

**Adaptation.** QuixBugs ships 40 single-function Python programs, each as a
buggy version (``python_programs/``) and a corrected reference
(``correct_python_programs/``), plus JSON test cases for most of them
(``json_testcases/``). Of the 40:

- **{len(_EXCLUDED)} were excluded outright** — not a difficulty judgement,
  only whether the program's argument/return shape can be represented and
  compared the way this project's sandbox harness does
  (``function(*input) == expected_output``, over JSON-stable values):
{exclusion_lines}
- **{candidates} candidates** remained.
- **{kept} of those** passed this project's own sandbox validator (the same
  ``validate_bug`` the v1/v2 dataset is checked with: the reference must pass
  every test case, the buggy version must fail at least one) after a real JSON
  round-trip. A program can also lose *some* (not all) of its published test
  cases here: a couple of QuixBugs' stress-sized cases (e.g. ``knapsack``'s
  multi-million-unit capacity, ``levenshtein``'s 50-character strings against
  an unmemoised recursive reference) cannot finish within any realistic sandbox
  time budget; those individual cases are dropped, keeping the rest of what the
  program exercises (a program is dropped entirely only if fewer than 3 cases
  survive that filter).

Each program's trailing docstring (QuixBugs embeds the problem statement and
doctest examples in the same file) is stripped; ``buggy_code``/``original_code``
contain only the function (plus any top-level imports it needs), matching every
other bug record in this project.

**What is intentionally missing.** ``bug_location.line_start`` is ``-1`` for
every record — QuixBugs does not publish a machine-readable line number for its
defects, and every localization-scoring path in this project already treats a
non-positive line as "cannot be credited" rather than as a wrong answer, so
this only zeroes the localization component, it does not break scoring.

**How to load it.** This set is *not* wired into ``load_bugs``/``load_tier``
(``difficulty`` is ``0``, outside the internal 1/2/3 tier scheme) and is never
trained on. Load ``bugs.jsonl`` directly, the same way ``scripts/gate_api.py``
loads a v2 directory with ``--bugs-dir``, or with
``Bug.from_dict(json.loads(line))`` per line.

**Counts.** {len(_EXCLUDED)} excluded, {candidates} candidates, {kept} validated
(kept/candidates = {kept / candidates:.0%}).
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
