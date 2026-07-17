"""Build a v2 bug dataset from MBPP by principled AST mutation.

Why this exists
---------------
The shipped 90-bug set is hand-templated mutations of canonical LeetCode problems
(``binary_search`` with ``right = mid + 1 + 1 + 1``, ``two_sum``, ``merge_sorted``).
A reviewer reads that as "template-based mutation of memorised problems", and a
strong hosted model solves ~all of it. This script replaces it with bugs sourced
from real, less-canonical problems (MBPP) and injected by a small taxonomy of
realistic mutation operators, each tagged with a difficulty tier and a bug type.

Method
------
1. Source problems from MBPP (reference solution + literal assert tests).
2. Parse each ``assert func(args) == expected`` into an (input, expected) case,
   keeping only JSON-stable, non-float expected outputs so the record survives a
   JSONL round-trip and the sandbox ``==`` comparison.
3. Inject exactly one bug with a tiered AST mutation operator:
     tier 1  boundary / off-by-one   (<-><=, >->>=, int constant +-1)
     tier 2  wrong operator / logic  (+/-, *//, and/or, ==/!=, </>)
     tier 3  edge-case               (slice bound, range stop, removed base case)
4. Validate every generated bug through the *existing* sandbox validator
   (reference passes all cases, buggy fails at least one), after a real JSON
   round-trip, so what is written is exactly what will later load.

Output: ``data/v2/bugs_tier{1,2,3}.jsonl`` plus ``data/v2/DATACARD.md``.

Usage:
    python scripts/build_dataset.py --per-tier 60
    python scripts/build_dataset.py --per-tier 60 --seed 7 --out data/v2
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# MBPP solutions contain regex string literals (``"\d"``) that trip SyntaxWarning
# when parsed; they are the sourced code's, not ours, and harmless here.
warnings.filterwarnings("ignore", category=SyntaxWarning)

from agentdebugger.dataset.models import Bug
from agentdebugger.dataset.validate import validate_bug
from agentdebugger.sandbox.policy import BLOCKED_IMPORTS, SandboxPolicy, analyze

# Modules that are importable but make tests non-deterministic; a bug over these
# would fail validation intermittently, so the problems are skipped outright.
_NONDETERMINISTIC = frozenset({"random", "time", "datetime", "secrets", "uuid"})
_MAX_CASES = 6
_MIN_CASES = 3


# ── sourcing ────────────────────────────────────────────────────────────────


def json_stable(value: Any, *, allow_float: bool) -> bool:
    """True if ``value`` round-trips through JSON without changing type/identity.

    Tuples become lists, sets are unserialisable, and floats break exact ``==``,
    so expected outputs must avoid all three; inputs may contain floats.
    """
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return allow_float
    if isinstance(value, str):
        return True
    if isinstance(value, list):
        return all(json_stable(v, allow_float=allow_float) for v in value)
    return False  # tuple, set, dict, bytes, ...


def parse_assert(source: str) -> tuple[str, tuple[Any, ...], Any] | None:
    """Parse ``assert func(<literals>) == <literal>`` into (name, args, expected)."""
    try:
        node = ast.parse(source.strip()).body[0]
    except (SyntaxError, IndexError):
        return None
    if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare):
        return None
    test = node.test
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return None

    left, right = test.left, test.comparators[0]
    call, expected_node = (left, right)
    if not isinstance(call, ast.Call):  # some asserts write `expected == func(...)`
        call, expected_node = right, left
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
        return None

    try:
        args = tuple(ast.literal_eval(a) for a in call.args)
        expected = ast.literal_eval(expected_node)
    except (ValueError, SyntaxError):
        return None

    if not all(json_stable(a, allow_float=True) for a in args):
        return None
    if not json_stable(expected, allow_float=False):
        return None
    return call.func.id, args, expected


@dataclass(frozen=True)
class Problem:
    """A sourced, cleaned problem: a reference solution and literal test cases."""

    task_id: str
    function_name: str
    code: str
    module: ast.Module
    cases: tuple[tuple[tuple[Any, ...], Any], ...]


def extract_problem(record: dict[str, Any]) -> Problem | None:
    """Turn a raw MBPP record into a :class:`Problem`, or ``None`` if unusable."""
    setup = (record.get("test_setup_code") or "").strip()
    code = ((setup + "\n\n") if setup else "") + record["code"].strip()

    try:
        module = ast.parse(code)
    except SyntaxError:
        return None

    imports = _imported_roots(module)
    if imports & (BLOCKED_IMPORTS | _NONDETERMINISTIC):
        return None
    if analyze(code, SandboxPolicy()):  # reference itself violates the policy
        return None

    parsed = [p for line in record.get("test_list", []) if (p := parse_assert(line))]
    if not parsed:
        return None

    names = [name for name, _, _ in parsed]
    function_name = max(set(names), key=names.count)  # the function the tests call
    if not _defines(module, function_name):
        return None

    cases = tuple((args, exp) for name, args, exp in parsed if name == function_name)
    cases = _dedupe_cases(cases)[:_MAX_CASES]
    if len(cases) < _MIN_CASES:
        return None

    return Problem(str(record["task_id"]), function_name, code, module, cases)


def _imported_roots(module: ast.Module) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".")[0])
    return roots


def _defines(module: ast.Module, name: str) -> bool:
    return any(
        isinstance(n, ast.FunctionDef) and n.name == name for n in ast.walk(module)
    )


def _dedupe_cases(cases):
    seen, unique = set(), []
    for args, exp in cases:
        key = json.dumps([args, exp], default=str)
        if key not in seen:
            seen.add(key)
            unique.append((args, exp))
    return unique


# ── mutation ────────────────────────────────────────────────────────────────

# A mutation operator flips one AST node in place. Each is tagged with a tier and
# a bug_type; the engine applies exactly one, at one site, per generated bug.

_CMP_FLIP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt}
_CMP_REVERSE = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE}
_EQ_FLIP = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_ARITH_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv}
_BOOL_SWAP = {ast.And: ast.Or, ast.Or: ast.And}


@dataclass(frozen=True)
class Operator:
    tier: int
    bug_type: str
    apply: Callable[[ast.AST], bool]  # mutate the node in place; return success


def _flip_compare_boundary(node: ast.AST) -> bool:
    if isinstance(node, ast.Compare) and type(node.ops[0]) in _CMP_FLIP:
        node.ops[0] = _CMP_FLIP[type(node.ops[0])]()
        return True
    return False


def _off_by_one_const(node: ast.AST) -> bool:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        node.value = node.value + random.choice((-1, 1))
        return True
    return False


def _swap_arith(node: ast.AST) -> bool:
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_SWAP:
        node.op = _ARITH_SWAP[type(node.op)]()
        return True
    return False


def _swap_bool(node: ast.AST) -> bool:
    if isinstance(node, ast.BoolOp) and type(node.op) in _BOOL_SWAP:
        node.op = _BOOL_SWAP[type(node.op)]()
        return True
    return False


def _flip_equality(node: ast.AST) -> bool:
    if isinstance(node, ast.Compare) and type(node.ops[0]) in _EQ_FLIP:
        node.ops[0] = _EQ_FLIP[type(node.ops[0])]()
        return True
    return False


def _reverse_compare(node: ast.AST) -> bool:
    if isinstance(node, ast.Compare) and type(node.ops[0]) in _CMP_REVERSE:
        node.ops[0] = _CMP_REVERSE[type(node.ops[0])]()
        return True
    return False


def _tweak_slice(node: ast.AST) -> bool:
    if isinstance(node, ast.Slice):
        for bound_name in ("lower", "upper"):
            bound = getattr(node, bound_name)
            if bound is None:
                continue
            delta = random.choice((-1, 1))
            if isinstance(bound, ast.Constant) and isinstance(bound.value, int):
                bound.value += delta  # fold, so `a[0:n]` -> `a[1:n]`, not `a[0 + 1:n]`
            else:
                setattr(node, bound_name, ast.BinOp(bound, ast.Add(), ast.Constant(delta)))
            return True
    return False


def _tweak_range_stop(node: ast.AST) -> bool:
    """`range(n)` -> `range(n - 1)`: an off-by-one that drops the last iteration."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
        if not node.args:
            return False
        stop_index = 1 if len(node.args) >= 2 else 0
        stop = node.args[stop_index]
        delta = random.choice((-1, 1))
        if isinstance(stop, ast.Constant) and isinstance(stop.value, int):
            stop.value += delta
        else:
            node.args[stop_index] = ast.BinOp(stop, ast.Add(), ast.Constant(delta))
        return True
    return False


_OPERATORS = (
    Operator(1, "comparison_boundary", _flip_compare_boundary),
    Operator(1, "off_by_one", _off_by_one_const),
    Operator(2, "wrong_arithmetic_operator", _swap_arith),
    Operator(2, "wrong_boolean_operator", _swap_bool),
    Operator(2, "flipped_equality", _flip_equality),
    Operator(2, "reversed_comparison", _reverse_compare),
    Operator(3, "off_by_one_slice", _tweak_slice),
    Operator(3, "off_by_one_range", _tweak_range_stop),
)


def _entry_nodes(module: ast.Module, function_name: str) -> list[ast.AST]:
    """Every node inside the entry function (so mutations stay localised to it)."""
    fn = next(
        n for n in ast.walk(module)
        if isinstance(n, ast.FunctionDef) and n.name == function_name
    )
    return list(ast.walk(fn))


def _remove_base_case(module: ast.Module, function_name: str) -> ast.Module | None:
    """Delete a leading ``if ...: return ...`` guard — a classic edge-case bug."""
    clone = copy.deepcopy(module)
    fn = next(
        n for n in ast.walk(clone)
        if isinstance(n, ast.FunctionDef) and n.name == function_name
    )
    for i, stmt in enumerate(fn.body):
        if (
            isinstance(stmt, ast.If)
            and len(stmt.body) == 1
            and isinstance(stmt.body[0], ast.Return)
            and not stmt.orelse
            and len(fn.body) > 1
        ):
            del fn.body[i]
            return clone
    return None


def mutate(problem: Problem, tier: int) -> tuple[str, str, str] | None:
    """Return (buggy_code, bug_type, canonical_original) for ``tier``, or None.

    Tries every operator/site for the tier in random order and returns the first
    mutation that parses and *changes behaviour*; behaviour is confirmed later by
    the sandbox validator.
    """
    original_code = ast.unparse(problem.module)
    attempts: list[tuple[str, ast.Module]] = []

    if tier == 3:  # structural operator has no per-node site list
        removed = _remove_base_case(problem.module, problem.function_name)
        if removed is not None:
            attempts.append(("removed_base_case", removed))

    node_count = len(_entry_nodes(problem.module, problem.function_name))
    for op in (o for o in _OPERATORS if o.tier == tier):
        for site in range(node_count):
            clone = copy.deepcopy(problem.module)
            nodes = _entry_nodes(clone, problem.function_name)
            if site < len(nodes) and op.apply(nodes[site]):
                attempts.append((op.bug_type, clone))

    random.shuffle(attempts)
    for bug_type, mutated in attempts:
        try:
            buggy_code = ast.unparse(mutated)
        except (ValueError, AttributeError):
            continue
        if buggy_code != original_code:
            return buggy_code, bug_type, original_code
    return None


def _first_changed_line(original: str, buggy: str) -> int:
    orig_lines = original.splitlines()
    for i, line in enumerate(buggy.splitlines(), start=1):
        if i > len(orig_lines) or line != orig_lines[i - 1]:
            return i
    return 1


def _initial_error(function_name: str, cases, buggy_code: str) -> str:
    """A short, honest failure description for the first failing case."""
    bug = _make_bug("probe", 1, "probe", function_name, buggy_code, buggy_code, "", 1, cases)
    from agentdebugger.sandbox import run_test_cases
    from agentdebugger.dataset.validate import _VALIDATION_POLICY

    results = run_test_cases(
        buggy_code, function_name, [c.as_dict() for c in bug.test_cases], policy=_VALIDATION_POLICY
    )
    for (args, expected), ok in zip(cases, results.outcomes, strict=False):
        if not ok:
            arglist = ", ".join(repr(a) for a in args)
            return f"Test failed: {function_name}({arglist}) should return {expected!r}."
    return "Some tests are failing."


def _make_bug(bug_id, tier, bug_type, function_name, buggy_code, original_code, error, line, cases):
    return Bug.from_dict(
        {
            "id": bug_id,
            "difficulty": tier,
            "bug_type": bug_type,
            "function_name": function_name,
            "buggy_code": buggy_code,
            "original_code": original_code,
            "initial_error": error,
            "bug_location": {"function": function_name, "line_start": line},
            "test_cases": [{"input": list(a), "expected_output": e} for a, e in cases],
        }
    )


def build_bug(problem: Problem, tier: int, bug_id: str) -> dict[str, Any] | None:
    """Generate one validated bug record for ``problem`` at ``tier``, or None."""
    result = mutate(problem, tier)
    if result is None:
        return None
    buggy_code, bug_type, original_code = result
    line = _first_changed_line(original_code, buggy_code)
    error = _initial_error(problem.function_name, problem.cases, buggy_code)

    bug = _make_bug(
        bug_id, tier, bug_type, problem.function_name,
        buggy_code, original_code, error, line, problem.cases,
    )

    # The record is only accepted if it survives a real JSON round-trip and the
    # existing sandbox validator: reference passes all, buggy fails at least one.
    record = bug.as_dict()
    roundtripped = Bug.from_dict(json.loads(json.dumps(record)))
    if not validate_bug(roundtripped).ok:
        return None
    return record


# ── driver ──────────────────────────────────────────────────────────────────


def load_mbpp_problems() -> list[Problem]:
    from datasets import concatenate_datasets, load_dataset

    dataset = load_dataset("google-research-datasets/mbpp", "full")
    combined = concatenate_datasets([dataset[split] for split in dataset])
    problems, seen = [], set()
    for record in combined:
        problem = extract_problem(record)
        if problem and problem.code not in seen:
            seen.add(problem.code)
            problems.append(problem)
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-tier", type=int, default=60, help="target bugs per tier")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/v2", help="output directory")
    args = parser.parse_args(argv)

    random.seed(args.seed)
    print("Loading MBPP and cleaning problems...", flush=True)
    problems = load_mbpp_problems()
    print(f"  {len(problems)} usable problems sourced.\n", flush=True)

    random.shuffle(problems)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    counts = {1: 0, 2: 0, 3: 0}
    used: set[str] = set()  # one bug per problem, to keep the pool diverse
    buckets: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: []}

    for tier in (1, 2, 3):
        for problem in problems:
            if counts[tier] >= args.per_tier:
                break
            if problem.task_id in used:
                continue
            bug_id = f"v2_t{tier}_{counts[tier] + 1:03d}"
            record = build_bug(problem, tier, bug_id)
            if record is not None:
                buckets[tier].append(record)
                counts[tier] += 1
                used.add(problem.task_id)
                print(f"\r  tier {tier}: {counts[tier]}/{args.per_tier}", end="", flush=True)
        print()

    for tier in (1, 2, 3):
        path = out / f"bugs_tier{tier}.jsonl"
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in buckets[tier]), encoding="utf-8"
        )
        print(f"  wrote {counts[tier]} bugs to {path}")

    total = sum(counts.values())
    _write_datacard(out, counts, args.seed)
    print(f"\nDone: {total} validated bugs in {out}/ (tiers {counts}).")
    if total < 3 * args.per_tier:
        print("Note: some tiers fell short; rerun with a different --seed to top up.")
    return 0


def _write_datacard(out: Path, counts: dict[int, int], seed: int) -> None:
    (out / "DATACARD.md").write_text(
        f"""# AgentDebuggerEnv bug dataset v2

**Source.** Problems sourced from MBPP (Mostly Basic Python Problems), reference
solutions and their literal `assert` tests. Less canonical than the v1
LeetCode-style set, to reduce memorisation.

**Bug injection.** Exactly one bug per record, introduced by a single AST
mutation operator, tagged with a difficulty tier:

- **Tier 1 — boundary / off-by-one:** comparison boundary flips (`<`↔`<=`,
  `>`↔`>=`) and integer-constant ±1.
- **Tier 2 — wrong operator / logic:** arithmetic swaps (`+`↔`-`, `*`→`//`),
  boolean swaps (`and`↔`or`), equality flips (`==`↔`!=`), comparison reversal.
- **Tier 3 — edge case:** slice-bound tweaks and removed base-case guards.

Tiers reflect the *mutation operator category*, not measured difficulty. Verify
that tier tracks difficulty empirically via a base model's per-tier solve rate
before making any curriculum claim (see docs/research_plan.md).

**Validation.** Every record passed `agentdebugger`'s sandbox validator after a
JSON round-trip: the reference solution passes every test case and the buggy
code fails at least one. Expected outputs are JSON-stable and non-float so the
sandbox `==` comparison is exact.

**Counts.** tier 1: {counts[1]}, tier 2: {counts[2]}, tier 3: {counts[3]}
(total {sum(counts.values())}). Generation seed: {seed}.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
