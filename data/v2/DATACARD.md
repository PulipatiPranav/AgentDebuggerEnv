# AgentDebuggerEnv bug dataset v2

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

**Counts.** tier 1: 60, tier 2: 60, tier 3: 60
(total 180). Generation seed: 0.
