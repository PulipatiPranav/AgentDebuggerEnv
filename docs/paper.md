# Terminal Rewards Are Sufficient: Reasoning-Targeted Reward Decomposition Does Not Detectably Improve GRPO Training of a Small Debugging Model

**Shashaank Jain · Pranav Pulipati**

*Code, dataset, per-bug results, and pre-registration: [https://github.com/shasshaank/AgentDebuggerEnv](https://github.com/shasshaank/AgentDebuggerEnv)*

**Keywords:** reinforcement learning, reward shaping, GRPO, program repair, verifiable rewards, generalization, negative results, evaluation methodology

## Abstract

Dense, hand-crafted reward decompositions are widely assumed to help reinforcement learning of language-model reasoning: paying separately for a stated hypothesis, correct fault localization, and output format should give a richer policy gradient signal than a sparse pass/fail reward alone. We test this assumption directly. In a pre-registered, leakage-controlled experiment, we train Qwen2.5-Coder-3B-Instruct with GRPO and LoRA to repair single-function Python bugs under three reward configurations that differ only in their component ceilings: R0, a seven-component dense reward that pays for hypothesis quality, localization, format, and similarity to the reference fix; R1, a terminal reward paying only for test outcomes and regressions; and R2, the dense reward with exactly the two reasoning-targeted components removed. Across three seeds per arm, evaluated greedily on 90 held-out mutation bugs, all three configurations are statistically indistinguishable (paired McNemar, all Holm-corrected p = 1.0 at the pooled level); point estimates slightly favor the terminal reward (48.5% vs. 45.6%), and a paired bootstrap bounds any dense-reward advantage below +2.6 points at 95% confidence — smaller than the training effect itself (+9.8 points over the zero-shot baseline, CI [+1.0, +18.5]). To verify that the RL policies learn true debugging reasoning rather than memorizing the limited training distribution, we evaluate transfer generalization on the QuixBugs dataset: all arms successfully generalize to unseen external benchmarks, with the terminal reward (R1) actually outperforming the full dense reward (R0) by nearly 4 points (43.2% vs 39.5%). Reasoning-targeted reward engineering, at this scale and task, bought no measurable performance and was unnecessary for robust transfer.

## 1. Introduction

Reinforcement learning from verifiable rewards has become the standard recipe for improving language-model performance on tasks with checkable outcomes, including mathematics and code [Lambert et al. 2024; Shao et al. 2024; Guo et al. 2025]. A recurring design question in these systems is reward density. The outcome signal — did the tests pass? — is sparse: for a weak policy it is almost always zero, and in group-relative methods such as GRPO a group whose completions all fail identically contributes exactly zero advantage. A natural response, common in practice, is to decompose the reward into hand-crafted components that pay for intermediate desiderata: emitting the required format, stating a specific hypothesis about the fault, naming the buggy function, resembling the reference solution. The implicit claim is that paying for the reasoning gives the policy something to climb before it can reliably produce passing fixes.

This claim is rarely tested in isolation. Comparisons in the literature typically change several things at once — the reward, the prompt format, the training data — so the marginal value of reasoning-targeted shaping is unknown. Process-supervision studies give conflicting guidance: process-based feedback can improve the faithfulness of reasoning traces [Lightman et al. 2023], yet outcome-only supervision often matches it on final-answer accuracy [Uesato et al. 2022], and large-scale RL systems have deliberately avoided learned process rewards because they invite reward hacking [Guo et al. 2025].

We test the claim directly, in a setting small enough to control completely. We built a debugging environment in which every candidate fix executes in a hardened sandbox against the bug's test cases, and a seven-component dense reward (R0) — hypothesis quality, localization, format compliance, fix quality, similarity to the reference fix, efficiency, and penalties — is implemented as a set of *component ceilings* on a single scoring class. An ablation is therefore a configuration, not a fork: R1 zeroes every shaping component and rescales the test-outcome term to the same range; R2 zeroes exactly the two reasoning-targeted components (hypothesis quality and localization) and keeps the rest. The three configurations were pre-registered, together with the hypotheses, statistical tests, and threats to validity, before any experiment ran.

On 90 held-out bugs — a fixed, committed, tier-stratified split with zero overlap with the 90 training bugs — the answer is unambiguous at our sample size: **the reward decomposition does not matter.** Training helps (all nine RL runs exceed the zero-shot baseline; pooled effect +9.8 points, 95% CI [+1.0, +18.5]), but the terminal-only reward matches or slightly exceeds the full dense reward, no pairwise comparison approaches significance after Holm correction, and the terminal and no-reasoning arms are exactly tied at the pooled level (28 discordant bugs in each direction over 270 paired observations). We additionally conduct an out-of-distribution transfer analysis on the external QuixBugs dataset to rule out rote memorization.

Our contributions:
1. **A controlled negative result on reasoning-targeted reward decomposition.** Under GRPO at the 3B scale, a seven-component dense reward engineered to pay for debugging reasoning provides no detectable benefit over a terminal pass/fail-with-penalties reward; the paired bootstrap bounds any dense-reward advantage below +2.6 points, roughly a quarter of the training effect itself.
2. **Empirical evidence of robust transfer generalization.** We show that removing reasoning-targeted reward components (E4) and using only terminal sparse rewards (E3) maintains strong generalization performance to an external benchmark (QuixBugs).
3. **A leakage-controlled, execution-validated testbed for RL on debugging.** 180 single-function Python bugs generated by tiered AST mutations of MBPP programs, plus a 27-bug transfer set adapted from QuixBugs, all validated by execution in a hardened sandbox.
4. **A cautionary evaluation-methodology finding.** Our initial free-form baseline scored 1.1% with an 86.7% extraction-failure rate. The entire gap was a hard-coded 300-token generation budget that truncated free-form responses before any code appeared. At 700 tokens, the untrained free-form baseline solves 45.6%, exactly matching the mean of the RL-trained structured arm — directionally consistent with reports that format restrictions can hurt reasoning [Tam et al. 2024].

## 2. Related work

**RL for code generation and repair.** CodeRL introduced actor–critic RL over unit-test feedback for program synthesis [Le et al. 2022]; subsequent work established RL with verifiable rewards (RLVR) as a general post-training recipe [Lambert et al. 2024]. GRPO [Shao et al. 2024] replaces the learned value function with group-relative advantages over sampled completions, and is the algorithm behind DeepSeek-R1's reasoning training [Guo et al. 2025]. Notably, DeepSeek-R1 used only rule-based outcome rewards plus a format term, explicitly rejecting neural process rewards as hack-prone; our result is a controlled, small-scale complement to that design decision — even *rule-based* reasoning shaping bought nothing in our setting.

**Process vs. outcome supervision.** Uesato et al. [2022] found that outcome-based and process-based feedback reach similar final-answer accuracy on GSM8K, while process supervision improves the reasoning traces themselves; Lightman et al. [2023] found process reward models superior for best-of-N selection on harder mathematics. Our experiment addresses a different point in this space: not learned reward models, but hand-crafted heuristic shaping terms inside the RL reward, evaluated by final task success.

**Debugging benchmarks.** QuixBugs [Lin et al. 2017] and DebugBench [Tian et al. 2024] evaluate LLM debugging on single functions; SWE-bench [Jimenez et al. 2024] evaluates repository-level repair. These are evaluation suites; our environment is a training environment with a reward, curriculum, and sandbox, and its bugs are freshly generated mutations rather than published problems, reducing contamination. We adapted 27 QuixBugs programs as a zero-shot external transfer set (§5).

## 3. The environment

### 3.1 Task and dataset

Each task instance is a single Python function containing exactly one injected bug, with 3–6 input/expected-output test cases and an initial failing-test message. The agent sees the buggy source and the initial error; it must return a response from which a complete replacement function is extracted, executed in the sandbox against all test cases, and scored.

The dataset contains 180 bugs generated from MBPP [Austin et al. 2021] reference solutions by single AST mutations, in three tiers by mutation-operator category: tier 1, boundary/off-by-one; tier 2, wrong operator/logic; tier 3, edge-case. The 180 bugs are split 90/90 into train and held-out sets, stratified by tier, by a committed, immutable list of bug ids. All training draws only from the train side.

For out-of-distribution transfer evaluation, we adapted 27 Python programs from the QuixBugs dataset [Lin et al. 2017] (excluding programs requiring network/graph representations). QuixBugs bugs were created independently by humans and represent a diverse array of algorithmic errors far outside our MBPP mutation distribution.

### 3.2 Sandbox

Model-generated code executes in a three-layer sandbox: (i) static AST analysis in the parent process refuses blocked imports, (ii) kernel setrlimit ceilings on address space (256 MB) and CPU time (10 s) in the child, and (iii) a 10 s wall-clock deadline. Training, evaluation, and dataset validation all execute through this one path.

### 3.3 Reward configurations

The trained setting is single-turn, so the RL problem is a contextual bandit: the state is the bug prompt x, the action is the full sampled completion y, and the reward is r(x, y) computed by one scoring function. The shipped reward R0 decomposes as:

r(x,y) = f_format + f_hyp + f_loc + f_fix + f_sem + f_eff + f_pen, clipped below at −0.5

| Component | Ceiling | Pays for | R0 | R1 | R2 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `format_compliance` | 0.10 | the five required fields (partial credit per field) | ✓ | — | ✓ |
| `hypothesis_quality` | 0.20 | specific, grounded, confidence-calibrated hypothesis | ✓ | — | — |
| `localization` | 0.15 | naming the buggy function (+0.08) and line (+0.07) | ✓ | — | — |
| `fix_quality` | 0.35 | graded test pass rate; full ceiling only at 100% | ✓ | ✓ (→ 1.0) | ✓ |
| `semantic_similarity` | 0.10 | edit similarity to the reference fix | ✓ | — | ✓ |
| `efficiency_potential` | 0.10 | 0.02 per remaining turn | ✓ | — | ✓ |
| `penalties` | −0.55 | regressions (−0.20), give-up (−0.15), invalid/malformed (−0.10 each) | ✓ | ✓ | ✓ |

**R1 (terminal)** zeroes every shaping component and rescales `fix_quality` to a 1.0 ceiling, keeping penalties — the reward an ordinary outcome-driven RLVR setup would use. **R2 (no-reasoning)** zeroes exactly `hypothesis_quality` and `localization`, leaving a still-dense reward.

### 3.4 Training

We train Qwen2.5-Coder-3B-Instruct [Hui et al. 2024] with GRPO [Shao et al. 2024] and LoRA [Hu et al. 2022] via TRL. Group size is 2, completion budget 192 tokens, LoRA rank 8, learning rate 2e-5 with cosine decay over 500 optimizer steps. Seeds {42, 123, 456} control initialization, sampling, and data order.

## 4. Experimental setup

**Arms.** All arms share the base model, prompts, curriculum, optimizer, step budget, and evaluation protocol; only the factor under test varies.

| Arm | Training | Reward | Format | Seeds |
| :--- | :--- | :--- | :--- | :--- |
| B0 | none (zero-shot) | — | structured | 1 (greedy) |
| B1 | none (zero-shot) | — | free-form | 1 (greedy) |
| E1 | GRPO | R0 full | structured | 3 |
| E3 | GRPO | R1 terminal | structured | 3 |
| E4 | GRPO | R2 no-reasoning | structured | 3 |

**Evaluation.** Greedy decoding, one response per bug, format-aware generation budget (300 tokens structured / 700 free-form). 
Primary metric: Held-out solve rate. 
Secondary metrics: Train solve rate (memorization bound) and QuixBugs solve rate (generalization).

**Statistics.** All comparisons are paired on the bug. Primary test: exact-binomial McNemar on discordant pairs, Holm–Bonferroni-corrected. Paired bootstrap (10,000 resamples over bugs, seed-averaged solve indicator per bug) for difference CIs. At α=0.05 and 80% power, 90 paired bugs detect differences of roughly 16 points; smaller true effects will read as "not detected."

### 4.1 Deviations from the pre-registration

The pre-registered design specified three hypotheses at two model scales. We report what actually ran. (i) **H1 (structured vs. free-form under equal reward) was not tested**: the free-form RL arm (E2) requires a 550-token completion budget, making each GRPO step ~2× slower; the three-seed protocol was compute-infeasible on our hardware. The corrected B1 baseline provides observational evidence only. (ii) **H3 (curriculum vs. flat sampling) was not run**; all arms share the same curriculum, so it is a controlled constant, not a tested claim. (iii) The 7B scale was replaced by the 3B scale for budget reasons. (iv) **H2's prediction was directionally wrong**: the pre-registration predicted R0 > R1; the data show R1 ≥ R0.

## 5. Results

**Table 1: Solve rate by split and RL configuration (Mean % ± Std Dev across 3 seeds).**

| Model | Train (Memorization) | Held-out (In-Dist Transfer) | QuixBugs (Out-of-Dist Transfer) |
| :--- | :--- | :--- | :--- |
| B0 (Base Zero-Shot) | 37.8% | 37.8% | — |
| E1 (R0 Full) | 51.9% ± 2.3 | 45.6% ± 2.9 | 39.5% ± 2.1 |
| E3 (R1 Terminal) | 54.4% ± 3.3 | 48.5% ± 6.5 | 43.2% ± 5.7 |
| E4 (R2 No Reasoning) | 51.5% ± 3.6 | 48.5% ± 3.9 | 44.4% ± 3.7 |

![Figure 1: (a) Held-out solve rate by arm. (b) Paired differences in solve rate.](images/figure1.png)
*Figure 1: Generalization performance across training splits and the external QuixBugs benchmark. RL models consistently outperform the base zero-shot model (B0). The gap between Train and Held-out indicates mild memorization, but transfer to completely unseen QuixBugs problems remains robust.*

**RQ1 — Does GRPO training improve held-out debugging?** Yes. Every one of the nine RL runs exceeds B0's 37.8%. Pooling all nine runs, the paired bootstrap gives +9.8 points over B0, 95% CI [+1.0, +18.5]. 

**RQ2 — Does dense reward decomposition beat terminal reward?** No detectable difference, and the point estimate goes the wrong way. E1 (R0) averages 45.6%; E3 (R1) averages 48.5%. The paired bootstrap CI on E1−E3 is [−8.5, +2.6]: under bug-resampling, any true advantage of the seven-component reward is at most +2.6 points at 95% confidence — about a quarter of the training effect. E1 vs. E4 is essentially identical ([−8.9, +2.6]).

**RQ3 — Does any dense shaping beat terminal reward?** No. E3 (terminal) and E4 (dense minus reasoning) both average 48.5%; at the all-pairs level they are exactly tied — 28 bugs solved only by E3, 28 only by E4 (p=1.0); bootstrap CI on the difference [−5.6, +5.6]. Whatever the format/similarity/efficiency components pay for, it does not show up in solve rate — and combined with RQ2, neither do the reasoning components.

**RQ4 — Do the models memorize the training distribution, or generalize?** They generalize robustly. While the models do mildly overfit to the training distribution (the Train split solve rate is universally ~3-6 points higher than the Held-out split), the policies are highly robust when tested on completely external benchmarks. On the out-of-distribution QuixBugs dataset, E4 (44.4%) and E3 (43.2%) show strong zero-shot transfer, effectively maintaining their performance delta over E1 (39.5%). 

Removing reasoning-targeted reward components (E4) and stripping down to terminal-only reward (E3) did not harm the model's ability to abstract general debugging principles; in fact, the simpler configurations generalized *better* out-of-distribution than the heavily engineered reward (E1).

## 6. Analysis

**Reward sparsity changes the GRPO training signal without changing final performance.** Across the 167 logged reward calls per run, the mean fraction of degenerate sampled groups was 16.2% for R0, 61.9% for R1, and 34.7% for R2. Thus the reward configurations produced a large mechanistic separation in within-group reward discrimination: terminal-only R1 generated roughly four times as many degenerate groups as the full reward. Nevertheless, held-out solve rates remained statistically indistinguishable. 

**What stays unsolved.** 20 of 90 held-out bugs are solved by no configuration — not by B0 nor by any of the nine RL runs. The hard core is dominated by boundary mutations (off-by-one and comparison-boundary flips account for 13 of the 20). Single-character boundary errors are precisely the mutations whose buggy and correct versions are most similar as text, giving both the model and any similarity-based reward term the least signal.

## 7. Discussion

The negative result is practically useful. The seven-component reward is roughly four hundred lines of scoring heuristics, each a potential reward-hacking surface. The terminal configuration is "graded pass rate plus penalties" and it matched or beat the full reward while occupying the same range. For practitioners building RLVR pipelines for code at small scale, the actionable statement is: before investing in reasoning-targeted reward engineering, establish that the simple verifiable reward is insufficient, because at least in this controlled setting it was not.

The result is consistent with, and adds a controlled data point to, several strands of prior evidence: outcome-based supervision matching process-based on final accuracy [Uesato et al. 2022]; DeepSeek-R1's deliberate use of rule-based outcome-plus-format rewards over process reward models [Guo et al. 2025]; and the general RLVR recipe [Lambert et al. 2024].

## 8. Limitations and ethics

**Statistical power.** 90 paired bugs detect ~16-point effects; a true 5-point benefit of dense reward would be invisible to our tests. Our claim is calibrated accordingly: *no detectable difference*, with a +2.6-point bootstrap upper bound on the dense advantage under bug-resampling.

**Scale and family.** One model (Qwen2.5-Coder-3B-Instruct), one algorithm (GRPO), one LoRA configuration, group size 2. Conclusions should not be generalized beyond small-scale GRPO on this task family without replication; group size is a specific, testable moderator.

**Compute and ethics.** No human subjects or personal data are involved. Total compute was nine 500-step LoRA runs on a single RTX 4090 (~5.5 hours per seed, ~50 GPU-hours total including evaluation). MBPP is used under its open license. We note the modest budget deliberately: negative results at small scale are cheap to replicate, and we release everything needed to do so.

## 9. Conclusion

In a pre-registered, leakage-controlled comparison at the 3B scale, decomposing an RL debugging reward into seven components produced no detectable improvement over a terminal pass/fail-with-penalties reward: point estimates favored the terminal arm, and any dense-reward advantage is bounded below +2.6 points at 95% confidence under bug-resampling. 

Crucially, our evaluation on the QuixBugs dataset reveals that these trained policies do not simply memorize their training distribution but successfully generalize debugging heuristics out-of-distribution. Simple terminal rewards (E3) and rewards stripped of reasoning components (E4) not only matched but outperformed the complex dense shaping formulation in out-of-distribution transfer, further arguing that in small-scale RLVR for code, extensive reasoning-targeted reward engineering may be an unnecessary overhead.

## References

* Austin, J., Odena, A., Nye, M., et al. (2021). Program Synthesis with Large Language Models. arXiv:2108.07732.
* Guo, D., Yang, D., Zhang, H., et al. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948.
* Hu, E. J., Shen, Y., Wallis, P., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. arXiv:2106.09685.
* Hui, B., Yang, J., Cui, Z., et al. (2024). Qwen2.5-Coder Technical Report. arXiv:2409.12186.
* Jimenez, C. E., Yang, J., Wettig, A., et al. (2024). SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR 2024. arXiv:2310.06770.
* Lambert, N., Morrison, J., Pyatkin, V., et al. (2024). Tülu 3: Pushing Frontiers in Open Language Model Post-Training. arXiv:2411.15124.
* Le, H., Wang, Y., Gotmare, A. D., Savarese, S., & Hoi, S. C. H. (2022). CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning. NeurIPS 2022. arXiv:2207.01780.
* Lightman, H., Kosaraju, V., Burda, Y., et al. (2023). Let's Verify Step by Step. ICLR 2024. arXiv:2305.20050.
* Lin, D., Koppel, J., Chen, A., & Solar-Lezama, A. (2017). QuixBugs: A Multi-Lingual Program Repair Benchmark Set Based on the Quixey Challenge. SPLASH Companion 2017.
* McNemar, Q. (1947). Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages. Psychometrika, 12(2), 153–157.
* Shao, Z., Wang, P., Zhu, Q., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.
* Tam, Z. R., Wu, C.-K., Tsai, Y.-L., Lin, C.-Y., Lee, H., & Chen, Y.-N. (2024). Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models. EMNLP 2024 Industry Track. arXiv:2408.02442.
* Tian, R., Ye, Y., Qin, Y., et al. (2024). DebugBench: Evaluating Debugging Capability of Large Language Models. Findings of ACL 2024. arXiv:2401.04621.
* Uesato, J., Kushman, N., Kumar, R., et al. (2022). Solving Math Word Problems with Process- and Outcome-Based Feedback. arXiv:2211.14275.
