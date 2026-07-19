"""Dense per-turn reward for a structured debugging response.

This is the signal GRPO optimises. It is deliberately *dense*: a purely
execution-based reward (did the tests pass?) is nearly always zero for a weak
model, which gives the policy nothing to climb. So the total is decomposed into
six earned components and one penalty term:

======================  =======  ==================================================
component               max      what it pays for
======================  =======  ==================================================
format_compliance       0.10     emitting the five required fields at all
hypothesis_quality      0.20     a specific, calibrated, non-generic hypothesis
localization            0.15     naming the function and line that actually broke
fix_quality             0.35     the fix passing the bug's test cases
semantic_similarity     0.10     the fix resembling the canonical one
efficiency_potential    0.10     solving early, while turns remain
penalties              -0.55     giving up, breaking tests, malformed output
======================  =======  ==================================================

The earned components sum to exactly 1.0 on a perfect first-turn solve, and the
total is floored at -0.5. Both bounds are covered by tests, because the
advertised reward range depends on them.

Deliberate design choice: ``fix_quality`` is the single largest component, so a
model cannot out-earn a real fix by writing beautiful prose about one.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from agentdebugger.protocol import StructuredAgentOutput

#: Turns an agent gets per episode in the structured setting.
MAX_TURNS = 5

#: Total reward can never fall below this, however badly a turn goes.
REWARD_FLOOR = -0.5


@dataclass(frozen=True)
class RewardBreakdown:
    """The itemised reward for one turn. ``total`` is the sum, floored."""

    format_compliance: float
    hypothesis_quality: float
    localization: float
    fix_quality: float
    semantic_similarity: float
    efficiency_potential: float
    penalties: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class GroundTruth:
    """What the reward function knows about the bug being fixed."""

    bug_function: str = ""
    bug_line: int = -1
    bug_type: str = ""
    canonical_fix_code: str = ""

    @classmethod
    def from_bug(cls, bug: Mapping[str, Any]) -> GroundTruth:
        """Build ground truth from a dataset bug record."""
        location = bug.get("bug_location") or {}
        return cls(
            bug_function=location.get("function", ""),
            bug_line=location.get("line_start", -1),
            bug_type=bug.get("bug_type", ""),
            canonical_fix_code=bug.get("original_code", ""),
        )


class TurnRewardCalculator:
    """Scores one structured response against the bug it was meant to fix.

    The component ceilings are constructor arguments, not fixed constants, so an
    ablation is a set of ceilings rather than a fork of the scoring code. The
    class-level defaults are the shipped **R0** reward; :meth:`terminal` (R1) and
    :meth:`no_reasoning` (R2) are the two ablations the experiment plan calls for.
    Setting a ceiling to ``0.0`` removes that component entirely — the earned
    credit *and* any partial/negative signal it would otherwise contribute.
    """

    # Default component ceilings — the shipped R0 reward. Changing one changes the
    # advertised reward range.
    FORMAT_MAX = 0.10
    HYPOTHESIS_MAX = 0.20
    LOCALIZATION_MAX = 0.15
    FIX_MAX = 0.35
    SEMANTIC_MAX = 0.10
    EFFICIENCY_PER_REMAINING_TURN = 0.02

    #: A fix that scores at least this much (the fix ceiling) has solved the bug.
    SOLVED_THRESHOLD = FIX_MAX

    def __init__(
        self,
        max_turns: int = MAX_TURNS,
        *,
        format_max: float = FORMAT_MAX,
        hypothesis_max: float = HYPOTHESIS_MAX,
        localization_max: float = LOCALIZATION_MAX,
        fix_max: float = FIX_MAX,
        semantic_max: float = SEMANTIC_MAX,
        efficiency_per_remaining_turn: float = EFFICIENCY_PER_REMAINING_TURN,
    ) -> None:
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")
        self.max_turns = max_turns
        # Instance ceilings shadow the class defaults, so existing references to
        # e.g. ``calculator.FIX_MAX`` keep working and read the configured value.
        self.FORMAT_MAX = format_max
        self.HYPOTHESIS_MAX = hypothesis_max
        self.LOCALIZATION_MAX = localization_max
        self.FIX_MAX = fix_max
        self.SEMANTIC_MAX = semantic_max
        self.EFFICIENCY_PER_REMAINING_TURN = efficiency_per_remaining_turn
        #: Solve detection tracks the fix ceiling, so it is correct under R1's
        #: rescaled fix reward as well as under R0/R2.
        self.SOLVED_THRESHOLD = fix_max

    # ── reward configurations (research_plan.md §3) ───────────────────────────

    @classmethod
    def full(cls, max_turns: int = MAX_TURNS) -> TurnRewardCalculator:
        """R0 — the full seven-component reward, exactly as shipped."""
        return cls(max_turns=max_turns)

    @classmethod
    def terminal(cls, max_turns: int = MAX_TURNS) -> TurnRewardCalculator:
        """R1 — monolithic terminal reward: fix outcome (rescaled to 1.0) + penalties.

        Every shaping component is removed; ``fix_quality`` is rescaled to a 1.0
        ceiling so a solve is still worth 1.0 and R0/R1 occupy the same range.
        """
        return cls(
            max_turns=max_turns,
            format_max=0.0,
            hypothesis_max=0.0,
            localization_max=0.0,
            semantic_max=0.0,
            efficiency_per_remaining_turn=0.0,
            fix_max=1.0,
        )

    @classmethod
    def no_reasoning(cls, max_turns: int = MAX_TURNS) -> TurnRewardCalculator:
        """R2 — dense reward with exactly the two reasoning components zeroed.

        Removes ``hypothesis_quality`` and ``localization`` only; format, fix,
        semantic and efficiency are untouched, so R2 is still a dense, climbable
        reward. R0 vs R2 asks whether paying for *reasoning* matters, or whether
        any dense shaping would do.
        """
        return cls(max_turns=max_turns, hypothesis_max=0.0, localization_max=0.0)

    @classmethod
    def from_name(cls, name: str, max_turns: int = MAX_TURNS) -> TurnRewardCalculator:
        """Build a calculator from a reward-config name (``R0``/``R1``/``R2``)."""
        builders = {
            "R0": cls.full,
            "full": cls.full,
            "R1": cls.terminal,
            "terminal": cls.terminal,
            "R2": cls.no_reasoning,
            "no_reasoning": cls.no_reasoning,
        }
        try:
            return builders[name](max_turns=max_turns)
        except KeyError:
            raise ValueError(
                f"Unknown reward config {name!r}. Choose from R0, R1, R2."
            ) from None

    def compute_turn_reward(
        self,
        agent_output: StructuredAgentOutput,
        ground_truth: GroundTruth,
        test_results: Mapping[str, Any],
        turn_number: int,
    ) -> RewardBreakdown:
        """Score a single turn."""
        proposing = agent_output.action == "propose_fix"

        format_compliance = self._format_score(agent_output)
        hypothesis_quality = self._hypothesis_score(agent_output, test_results, proposing)
        localization = self._localization_score(agent_output, ground_truth)
        fix_quality = self._fix_score(test_results, proposing)
        semantic_similarity = self._semantic_score(agent_output, ground_truth, proposing)
        efficiency = self.EFFICIENCY_PER_REMAINING_TURN * max(0, self.max_turns - turn_number)
        penalties = self._penalties(agent_output, test_results)

        total = (
            format_compliance
            + hypothesis_quality
            + localization
            + fix_quality
            + semantic_similarity
            + efficiency
            + penalties
        )

        return RewardBreakdown(
            format_compliance=round(format_compliance, 4),
            hypothesis_quality=round(hypothesis_quality, 4),
            localization=round(localization, 4),
            fix_quality=round(fix_quality, 4),
            semantic_similarity=round(semantic_similarity, 4),
            efficiency_potential=round(efficiency, 4),
            penalties=round(penalties, 4),
            total=round(max(total, REWARD_FLOOR), 4),
        )

    # ── components ────────────────────────────────────────────────────────────

    def _format_score(self, output: StructuredAgentOutput) -> float:
        """Full marks for a well-formed response; partial credit per field otherwise.

        The gradient matters: a model that emits three of five fields is closer
        to the target than one that emits prose, and the reward has to say so or
        it will never find the format by exploration.
        """
        if self.FORMAT_MAX == 0:  # component removed (R1)
            return 0.0
        if output.valid:
            return self.FORMAT_MAX

        fields_present = sum(
            (
                len(output.observation) > 5,
                len(output.hypothesis) > 10,
                output.confidence in {"low", "medium", "high"},
                output.action != "invalid",
                len(output.detail) > 0,
            )
        )
        return -0.25 + 0.04 * fields_present

    def _hypothesis_score(
        self,
        output: StructuredAgentOutput,
        test_results: Mapping[str, Any],
        proposing: bool,
    ) -> float:
        """Reward specific, grounded, calibrated hypotheses over generic ones."""
        if self.HYPOTHESIS_MAX == 0:  # component removed (R1, R2)
            return 0.0
        score = 0.0
        hypothesis = output.hypothesis

        if len(hypothesis.split()) >= 20:  # substantive, not a one-liner
            score += 0.05
        if re.search(r"[`'\"<>!=+\-*/]", hypothesis):  # cites code, an operator, a symbol
            score += 0.05
        if re.search(r"\d", hypothesis):  # cites a line number, an index, a bound
            score += 0.05
        if self._grounded_in_observation(output):
            score += 0.05

        # Confidence calibration: being sure and right beats being sure and wrong.
        if proposing:
            solved = _passed(test_results) == _total(test_results) and _total(test_results) > 0
            if output.confidence == "high":
                score += 0.05 if solved else -0.05
            elif output.confidence == "low" and solved:
                score += 0.02

        return max(0.0, min(score, self.HYPOTHESIS_MAX))

    @staticmethod
    def _grounded_in_observation(output: StructuredAgentOutput) -> bool:
        """True when the hypothesis reuses the vocabulary of what was observed."""
        observed = set(output.observation.lower().split())
        hypothesised = set(output.hypothesis.lower().split())
        if not observed:
            return False
        return len(observed & hypothesised) / len(observed) > 0.15

    def _localization_score(
        self, output: StructuredAgentOutput, ground_truth: GroundTruth
    ) -> float:
        """Reward naming where the bug is, not just what it does."""
        if self.LOCALIZATION_MAX == 0:  # component removed (R1, R2)
            return 0.0
        score = 0.0
        text = f"{output.hypothesis} {output.detail}".lower()

        function = ground_truth.bug_function.lower()
        if function and function in text:
            score += 0.08
        if ground_truth.bug_line > 0 and str(ground_truth.bug_line) in output.hypothesis:
            score += 0.07

        return min(score, self.LOCALIZATION_MAX)

    def _fix_score(self, test_results: Mapping[str, Any], proposing: bool) -> float:
        """Pay for tests that actually pass. Graded, so partial fixes still climb.

        Partial credit is expressed relative to the fix ceiling, so the graded
        shape is preserved when R1 rescales the ceiling from 0.35 to 1.0.
        """
        total = _total(test_results)
        if not proposing or total == 0:
            return 0.0

        pass_rate = _passed(test_results) / total
        scale = self.FIX_MAX / type(self).FIX_MAX  # 1.0 under R0/R2, 1/0.35 under R1
        if pass_rate == 1.0:
            return self.FIX_MAX
        if pass_rate >= 0.75:
            return 0.20 * scale
        if pass_rate >= 0.50:
            return 0.12 * scale
        if pass_rate > 0.0:
            return 0.05 * scale
        return 0.0

    def _semantic_score(
        self,
        output: StructuredAgentOutput,
        ground_truth: GroundTruth,
        proposing: bool,
    ) -> float:
        """A small nudge towards the canonical fix, for fixes that are close but not passing.

        Kept small on purpose: it is a similarity heuristic, not a correctness
        oracle, and it must never outweigh ``fix_quality``.
        """
        if self.SEMANTIC_MAX == 0:  # component removed (R1)
            return 0.0
        canonical = ground_truth.canonical_fix_code
        if not proposing or not output.detail or not canonical:
            return 0.0

        similarity = difflib.SequenceMatcher(None, output.detail, canonical).ratio()
        if similarity >= 0.85:
            return self.SEMANTIC_MAX
        if similarity >= 0.65:
            return 0.05
        if similarity >= 0.40:
            return 0.02
        return 0.0

    def _penalties(
        self, output: StructuredAgentOutput, test_results: Mapping[str, Any]
    ) -> float:
        """Price the behaviours the environment exists to discourage."""
        penalty = 0.0
        if int(test_results.get("newly_broken", 0)) > 0:
            penalty -= 0.20  # a fix that breaks passing tests is worse than no fix
        if output.action == "give_up":
            penalty -= 0.15
        if output.action == "invalid":
            penalty -= 0.10
        if not output.valid:
            penalty -= 0.10
        return penalty

    # ── episode-level aggregation ─────────────────────────────────────────────

    def compute_episode_reward(self, trajectory: Sequence[Mapping[str, Any]]) -> float:
        """Discounted sum of turn rewards, plus a bonus if the bug was ever solved.

        The discount (0.9 per turn) is what makes solving on turn 1 worth more
        than solving on turn 4 even though both end in a fix.
        """
        if not trajectory:
            return 0.0

        total = 0.0
        discount = 1.0
        for turn in trajectory:
            total += discount * turn["reward"].total
            discount *= 0.9

        if any(turn["reward"].fix_quality >= self.SOLVED_THRESHOLD for turn in trajectory):
            total += 0.20

        return round(total, 4)

    def mean_components(self, trajectory: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        """Per-component means over a trajectory, for logging."""
        if not trajectory:
            return {}

        components = [f.name for f in _BREAKDOWN_FIELDS if f.name != "total"]
        return {
            f"reward/{name}": round(
                sum(getattr(turn["reward"], name) for turn in trajectory) / len(trajectory), 4
            )
            for name in components
        }


_BREAKDOWN_FIELDS = tuple(RewardBreakdown.__dataclass_fields__.values())


def _passed(test_results: Mapping[str, Any]) -> int:
    return int(test_results.get("passed", 0))


def _total(test_results: Mapping[str, Any]) -> int:
    return int(test_results.get("total", 0))
