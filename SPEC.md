# Transformational Creativity for Autonomous Research


Build a system that helps research agents identify the assumptions organizing a problem, change a substantive assumption or representation, and test the resulting idea through small experiments. The aim is Boden’s transformational creativity: changing the rules of a conceptual space. The first milestone is a credible implemented reformulation, with evidence about whether it helps; historical novelty remains a separate judgment.

Use **K** for knowledge and working commitments, and **C** for their context and priors. A revisable dependency DAG represents this conceptual “box.” It includes known alternatives and distinguishes active commitments from background knowledge. It is a scientific dependency graph, not the candidate model’s computation graph. Its vocabulary can expand, and agents can reconstruct the formulation.

## 2. Two questions, one implementation

| Stage | Question | Deliverable |
| --- | --- | --- |
| **A. Core feasibility and smoke-test validation** | Can agents turn explicit assumptions into substantive, executable reformulations? Do cheap tests predict longer-run outcomes? | Working prototype, research cases, and paired short/long experimental results |
| **B. Exploratory representation comparison** | Does supplied DAG structure help beyond the same information in a flat register? | Three matched DAG–flat campaign pairs with descriptive comparisons |

Stage A uses the proposed DAG-based system. Stage B begins after its loop and tests work. A DAG–flat tie would not refute the shared core approach; a loss improvement alone would not establish transformational creativity. Establishing an advantage over ordinary autoresearch would require a later comparator.

## 3. First setting: small causal language-model architectures

Study **state representation and internal computation** in a domain the researcher understands. Start near 10–30M total parameters, subject to profiling, with a tested Llama-like decoder using RoPE, RMSNorm, and SwiGLU. Freeze the exact configuration, tokenizer, training recipe, and a modest tuning allowance before comparisons. Known component upgrades are ordinary changes, not automatic formulation changes.

Choose **one** language corpus and a relevant mechanism probe. For a memory-focused question, a frozen [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) subset plus controlled recall is a reasonable starting choice. Verify that context restriction or a known mechanism ablation affects the chosen evaluation. TinyStories remains an inexpensive debugging option; it is not a mandatory second benchmark. Do not require a generic recall improvement for an unrelated computation hypothesis.

Keep next-token prediction, causal prefix access, splits, evaluator, and resource limits fixed. Allow new state layouts, information pathways, primitives, or computation schedules. The interface must not force a particular internal representation. Compare against the starting decoder and the nearest relevant established mechanism.

## 4. Build and exercise the core loop

Start with a small source-grounded box reviewed with the mentor. Separate defining construction rules from routine hyperparameters. Conditional theoretical results retain their scopes; suspending an assumption does not falsify a theorem.

Use a small agent group, shared evidence, and a persistent archive:

1. Identify a commitment to question, or propose a different formulation.
2. Record the proposed mechanism, a discriminating prediction, and a control before testing.
3. Implement the idea and run engineering checks followed by a scientific smoke test.
4. Record observations, failures, costs, and unresolved questions; share them with the group.
5. Continue a bounded unresolved attempt, challenge another commitment, or reconstruct from observations in a fresh context.

Keep research-agent weights and orchestration fixed. Preserve GEA-inspired experience sharing and DGM-inspired branch persistence; defer policy evolution and an adaptive manager. The researcher and mentor inspect initial cases as ordinary exploration, known-space switching, substantive reformulation, or uncertain. Check whether the code actually expresses the claimed change. No graph-edit score substitutes for this judgment.

## 5. Make smoke-test fidelity a main pilot result

Separate **implementation validity**, **mechanism evidence**, and **early utility**. Enforce causal leakage checks and state resets. For scientific screening, measure improvement over the control at two predefined training-compute checkpoints at one model size. Keep the training schedule consistent with the planned longer run. Weak early loss can be inconclusive; it is not a universal rejection of an unfamiliar mechanism.

For calibration, enroll approximately **8–12 valid candidate implementations before their scientific smoke outcomes are known**, including a few known controls. Give every candidate an independent-seed longer run regardless of its early score. Freeze code and configuration across evaluations; a redesigned descendant is a different candidate. Keep longer results hidden until enrollment and short-test decisions close. Add seed repetitions for promising or ambiguous findings; report controls separately from agent discoveries.

Before longer results, freeze an advance/hold/reject rule and the task-relevant improvement/resource limits for provisional usefulness. Keep noisy or unreplicated cases uncertain. Produce:

- Paired early-versus-later performance plots and a descriptive rank correlation.
- A table of screening decisions versus later usefulness, including missed useful candidates.
- Explanations of failures, especially whether screening rejects substantive reformulations while favoring routine tuning.

This directly tests the proxy within the chosen regime. A small, related candidate batch does not establish a universal correlation or predict frontier scaling. The detailed appendix defines the comparison and denominators.

## 6. Then compare DAG and flat representations

Freeze the harness and run **three matched pairs of short campaigns**, with separate archives. Pair starting tasks, code, and training-seed schedules. Both conditions receive the same claims, source material, dependency statements, challenge prompts, group sharing, reconstruction opportunities, and budgets. The flat arm stores these in text; the DAG arm also receives structured dependencies and graph operations. Both may change the formulation.

Compare implemented changes, mechanism evidence, utility at a common training-compute checkpoint, costs, failure modes, and whether dependencies guided experiments before results. Nominate promising candidates using development evidence without requiring self-labels of creativity. Use concise condition-neutral case descriptions where practical. Six campaigns support a comparative pilot, not a reliable rare-success-rate claim. Freeze any screening revisions before Stage B, which uses fresh candidates and untouched confirmation cases.

## 7. Deliverables, limits, and mentor decisions

Deliver an executable prototype, an auditable case ledger, a smoke-test calibration report, and an exploratory DAG–flat comparison. Meaningful negative results include an uninformative proxy, frequent implementation failures, or reformulations that do not improve utility.

Before agent campaigns, profile the runner and complete agent episodes. Set caps for inference, training, confirmation, and researcher review. Match training compute, report inference cost and memory, and protect confirmation data. Expand the candidate batch or campaign count only within an agreed resource envelope.

**Mentor discussion:** agree on what counts as a defining commitment in the chosen architecture problem, whether the task exposes the proposed mechanism, and the role of the DAG comparison as a secondary design question. Begin with Stage A; continue to Stage B when the prototype produces interpretable evidence.

### Reading anchors

[Boden](https://doi.org/10.1016/S0004-3702(98)00055-1) and [Graphical Theory](https://arxiv.org/abs/2504.18687v2) motivate the conceptual box; [GEA](https://arxiv.org/abs/2602.04837v1) and [DGM](https://arxiv.org/abs/2505.22954v3) inform shared experience and persistence; [Ideation–Execution Gap](https://arxiv.org/abs/2506.20803v1) motivates execution; [MAD](https://arxiv.org/abs/2403.17844v2) and [Zoology](https://arxiv.org/abs/2312.04927) inform architecture probes.

**Companion:** the appendix below contains implementation notes, qualifications, full references, and the deferred confirmatory design.

---

## Transformational Creativity — Implementation and Later-Study Appendix
-
**Companion to the short pilot plan · November 2026**

The pilot plan is the primary project document. Sections A–D below clarify implementation and interpretation. Section E preserves decisions needed only for a stronger subsequent study; they are not prerequisites for the first prototype. This revision separates the core feasibility question from the incremental value of supplied graph structure.

### A. Minimal representation and group implementation

#### A1. The box

K contains definitions, assumptions, methods, empirical claims, and conditional theoretical results. C specifies context, priors, and permitted constructions. Mark which commitments define the current formulation and which are ordinary tunable choices. Include established alternatives so that a conventional starting model is not mistaken for the frontier of available knowledge.

An edge **A → B** means B’s recorded justification or construction uses A; this prerequisite-first convention reverses [2]. Represent joint/alternative support explicitly: `supports: [[A, B], [D]]` means “A and B together, or D.” Suspending support flags dependent claims for reconsideration; it does not establish their falsity. Changing C creates a new version. Inherited claims preserve their original scope, with applicability in the new context supported, conditional, or unresolved.

Agents may create primitives, replace subgraphs, or reconstruct the space. A first idea need not fit the old vocabulary; its experiment needs operational definitions. The graph records scientific justification, not model execution, so recurrent model computation is permitted. Archive ancestry is a separate relationship.

For Stage A, review a small source-based box with the mentor. For Stage B, freeze the extraction protocol and starting materials independently of which condition wins. Both arms receive the same substantive information, including dependencies in prose. The flat arm is free to infer its own graph; the estimand concerns supplied dependency management, not a ban on baseline reasoning.

#### A2. Small-group loop

Use four interchangeable job slots per wave: two challenges, one reconstruction, and one continuation or fallback challenge. Freeze a wave-start evidence snapshot; results become shared next wave. Pool observations, failures, and reusable components without forcing consensus [3, 4]. Reconstruction initially receives observations and the task contract, with interpretations, candidate code, and formulations available through logged retrieval [5].

The prototype’s two challenge parents are the development incumbent and a valid archived state selected by least-recent expansion, with ID tie-breaking. Initialize the research archive with the tested starting decoder; keep comparison controls in a separate registry. Set each state’s expansion clock to its creation wave and update it on parent assignment, not donor retrieval. Duplicate same-wave parent selections are allowed. Define the incumbent at the common development-compute checkpoint.

Record `research_parent_id`, `code_base_id`, `donor_ids`, `origin_id`, `action_mode`, and `evidence_snapshot_id`. Workers do not own lineages. A continuation becomes a child of the selected proposal regardless of which worker runs it. Reconstruction has no research parent and begins from a clean harness containing data/training/evaluation plumbing, without candidate architecture code or trained weights. Record subsequent imports as donors.

Challenges/reconstructions receive scheduler-assigned origins; continuations retain them. Each task-valid proposal may queue one precommitted unresolved follow-up. Dispatch removes it and consumes one of that origin’s two continuation allowances, even on failure or timeout. Workers cannot reset these counters. If no follow-up qualifies, that same slot becomes a new challenge. New challenges can reuse earlier code; the continuation cap is not a claim of semantic uniqueness.

These are simple implementation defaults, adjustable during Stage A and frozen before Stage B. Stage A’s predetermined calibration cohort receives its longer evaluations separately from this scheduler, regardless of screening outcomes.

#### A3. Minimal records and boundaries

Store versioned formulation, proposal, code/configuration hash, predictions, controls, observations, validity status, costs, and next question. JSON records plus a simple indexed artifact store suffice initially. Preserve failures and incomplete states without promising unlimited continuation.

The external evaluator owns data access, causal checks, metric calculation, and resource enforcement. Candidates receive equal causal prefixes; persistent state resets between independent documents and runs. Changing allowed history requires a separate task definition. Check full-sequence versus incremental predictions where both modes exist. Inherited trained or optimizer state requires explicit resumption accounting; usefulness confirmation uses fresh training runs.

### B. Architecture setting and smoke-test calibration

#### B1. Choose a task that exercises the mechanism

A small Llama-like baseline with RoPE, RMSNorm, and SwiGLU has a relevant precedent in MAD’s Transformer++ [10]. Freeze its implementation, tokenizer, context length, and tested training recipe. Embedding tying and QK normalization are recipe choices rather than universal upgrades. Give candidate families the same bounded tuning policy [12]. A strong baseline reduces easy rediscoveries but does not make every remaining improvement transformational.

Choose one corpus through known-control profiling. For longer-context memory questions, a fixed FineWeb-Edu subset [19] and a targeted recall/state diagnostic are a reasonable proposal. Verify sensitivity by restricting accessible context or ablating relevant state. Web text alone does not guarantee a memory-sensitive test.

TinyStories remains a possible small-scale domain [9]. Its paper reports architecture-dependent context tracking; Zoology does not test TinyStories. Zoology instead demonstrates gaps between earlier synthetic recall tests and language behavior [11]. Therefore the decision should depend on the mechanism and measurable task sensitivity, not an unsupported declaration that TinyStories cannot reveal state effects.

Keep language loss, resource cost, and mechanism-probe results separate. A required recall improvement is appropriate for a recall research question; it should not exclude unrelated computation proposals. Small artificial tasks with provably insufficient reference information access can calibrate the audit, but deliberately known solutions are not discovery evidence.

#### B2. What predicts what?

Enroll a small cohort by a fixed rule before each candidate’s scientific smoke outcomes—for example, the first eligible valid implementations up to the cap. Engineering-invalid attempts remain in the failure ledger. Known controls and agent-generated candidates are labeled separately. Every valid enrolled candidate gets short and longer evaluation; confirmation is not selected using the short score.

For each frozen candidate, measure development loss relative to its applicable control at training-compute checkpoints b1 and b2, plus the precommitted mechanism test. Compare these signals with an independent-seed run at a larger declared budget B and held-out cases. Short training uses the prefix of the planned full-horizon schedule, so a different decay schedule is not an accidental confound. This fresh-seed longer run is the default for every enrolled artifact; repeat promising or ambiguous findings with additional seeds. A future cheaper resumed-run study would answer trajectory prediction and must be labeled separately.

Changing the architecture, tuning, or data recipe after the smoke test creates a descendant. Its improvement may demonstrate successful refinement, but must not be counted as predictive accuracy for the frozen original artifact. Do not expose the calibration batch’s longer-run outcomes to agents until enrollment and short-test decisions have closed.

Before longer results, record a provisional advance/hold/reject rule and the minimum baseline-relative gain and resource limits for provisional usefulness. This pilot can use a descriptive point-estimate criterion, with uncertain/unreplicated cases labeled; it does not require the confirmatory machinery in §E. Report all paired points, learning-curve crossings, and descriptive rank correlations for comparable signals. A pooled correlation can conceal a proxy that ranks ordinary variants well but misses reformulations; inspect cases by type and provenance. Candidate descendants and common controls create dependence, so candidate count is not an independent-campaign sample size.

Report both screening errors with explicit denominators:

- **Missed-useful fraction:** useful candidates rejected / all useful candidates.
- **Useful-among-rejects fraction:** useful candidates rejected / all rejected candidates.

Show counts, and mark a zero denominator as undefined. With 8–12 candidates these describe the cohort, not precise population rates. If confirming the full cohort exceeds the cap, shrink it before enrollment; do not silently retain only winners. Refined screening rules learned from Stage A need fresh evaluation in Stage B.

Two compute checkpoints at one model size test learning dynamics. A second size addresses a different question—scale dependence—and is reserved for a specific hypothesis. Neither a smoke test nor a two-size trend establishes frontier scaling [10, 11].

### C. Exploratory DAG–flat comparison

Use three matched pairs of short campaigns after freezing the harness. Pair tasks, initial code, source materials, model configurations, seed schedules, and budgets. Separate archives and fresh contexts prevent cross-arm transfer. Share experience within each group. Charge graph queries, serialization, and maintenance to its arm; report one-time human preparation separately.

Rank candidates at the same development training-compute checkpoint. Insufficiently trained candidates remain pending. Use equal tuning-resource policies, report inference cost and memory, and count failures and repairs. Parameter equality is not compute equality for recurrent or adaptive architectures. Compare useful finalists with the nearest applicable established mechanism, selected before held-out results.

Use concise common-format case cards with the commitment, proposed mechanism, implementation evidence, outcomes, and scope. The researcher and mentor may inspect initial cases directly; condition-neutral descriptions improve the Stage B comparison where feasible. Do not imply fully independent blinded judging if that was not performed.

Report ordinary exploration, known-space switches, substantive reformulations, and uncertain cases separately from utility and novelty. Historical novelty requires literature comparison. Compare all campaign outcomes, dependency use before experiments, costs, and failures. A DAG–flat tie is inconclusive about graph benefit and does not refute the shared core; a DAG loss advantage alone does not demonstrate transformational creativity. Neither comparison establishes superiority over ordinary autoresearch without an appropriate additional arm.

### D. Cost and execution decisions

Profile known architecture controls and several complete agent episodes before setting cohort and campaign caps. Set budgets for agent inference, experimentation, longer confirmation, baseline tuning, and researcher review. The six Stage B campaigns are optional until Stage A produces interpretable evidence at an affordable cost; freeze their duration before running them.

Sum input and output usage across all tool-loop calls, including repeated context and repairs; distinguish cached inputs for billing. Record real GPU-hours as well as training FLOPs, inference latency, memory, and processed tokens. Equal FLOPs can entail unequal token exposure; report both.

The rough dense-model estimate 6NT gives 1.08×10^17 FLOPs for N=30M nonembedding dense parameters and T=600M tokens [20]. This N differs from the pilot’s total-parameter scale. The approximation omits some embedding/context-dependent costs and cannot be transferred unchanged to arbitrary recurrent execution. Finishing that estimate in 15–30 minutes would require approximately 120–60 sustained TFLOP/s. It is a throughput requirement, not a hardware-independent runtime prediction. Actual measurements determine the budget.

### E. Deferred confirmatory design

**Use this section only when seeking stronger comparative claims. It is not the pilot’s acceptance checklist.**

#### E1. Define the claim and comparator

Decide whether the study tests supplied graph management, explicit formulation work as a package, smoke-test selection, or another component. A flat-versus-DAG comparison isolates the first. A claim that the full approach improves ordinary autoresearch needs a separately designed baseline. Historical novelty, usefulness, and formulation change remain separate judgments.

Freeze source construction, tasks, nomination, tuning, utility thresholds, and analysis before confirmatory outcomes. Broaden beyond one hand-built box and one training regime before claiming generality. Heuresis, UoT, and EvoSCM are close prior work [6, 17, 18]; the contribution must be demonstrated rather than inferred from combining their ingredients.

#### E2. Nomination and useful reformulation

A possible endpoint is whether a campaign produces at least one useful, validated formulation change among up to three finalists. Nominate from all eligible proposals by development utility, with lower-cost/ID tie-breakers and exact artifact deduplication. Self-assigned creativity labels and graph edits do not determine eligibility. The endpoint concerns nominees, not every idea generated.

A qualifying change alters a defining representation or construction rule, appears in executable behavior, preserves the task, and satisfies the declared usefulness criterion. Tuning or renaming alone is insufficient. Freeze applicable controls and their configurations using development evidence before sealed confirmation.

For a loss-based claim of reliably meaningful improvement, require the prespecified lower confidence bound on control loss minus candidate loss to exceed the chosen practical threshold. Training seeds, not token-level sample size, capture training variability. Address up to three confirmation opportunities with an appropriate joint procedure. Include every launched campaign; report uncertain judgments and their implications.

#### E3. Review and holdouts

Use structured condition-neutral packets, code evidence, two independent domain reviewers, and adjudication when needed. Remove graph vocabulary and ancestry labels; record condition guesses. LLM judges may assist but should not alone determine the endpoint [16]. Performance-margin hiding limits halo effects, although usefulness-first screening is not complete outcome blinding.

Screened-out candidates have unassessed transformation status, not a negative label. A full small cohort avoids this problem in the pilot. Larger selectively reviewed studies need an explicit sampling/analysis plan before making population claims about transformation rates.

At fixed budget checkpoints, freeze the development-selected incumbent. Evaluate it on sealed data without returning scores or selecting held-out winners [13]. If development uses previously revealed outcomes, reserve fresh final cases. Report final utility alongside any utility-versus-budget curve.

#### E4. Sample size and review feasibility

Power the actual endpoint and meaningful effect. The illustrative 10% versus 30% independent campaign-success contrast needs approximately 62 campaigns per arm under a standard two-sided 5%/80%-power approximation; the pilot cannot estimate rare rates precisely. Ordinal diagnostics and continuous utility answer related but different questions. Predefine any multiple-endpoint interpretation rather than selecting a favorable result afterward.

Budget review explicitly. Sixty campaigns per arm, three nominees, and two reviewers can create 720 initial decisions; at an assumed 15–20 minutes each, 180–240 reviewer-hours before adjudication. Screening may reduce the work, but the pilot should measure review time. If a powered study is unaffordable, report a bounded exploratory finding rather than claiming equivalence from a null result.

### F. Related work and sources

1. Margaret A. Boden (1998). [Creativity and artificial intelligence](https://doi.org/10.1016/S0004-3702(98)00055-1). *Artificial Intelligence* 103(1–2), 347–356. Conceptual foundation.
2. Samuel Schapiro, Jonah Black, and Lav R. Varshney (2025). [Transformational Creativity in Science: A Graphical Theory](https://arxiv.org/abs/2504.18687v2). Dependency representation; graph-level potential does not establish usefulness.
3. Zhaotian Weng et al. (2026). [Group-Evolving Agents: Open-Ended Self-Improvement via Experience Sharing](https://arxiv.org/abs/2602.04837v1). Group experience inheritance; full framework evolution is deferred here.
4. Jenny Zhang et al. (2025; v3, 2026). [Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents](https://arxiv.org/abs/2505.22954v3). Persistent branching archive.
5. Ziyang Cai, Christos Ziakas, et al. (2026). [Fork-and-Flush: Escaping Idea Basins in Autoresearch Agents](https://arxiv.org/abs/2610.07447v1). Fresh contexts with workspace inheritance.
6. Antonis Antoniades et al. (2026). [Heuresis: Search Strategies for Autonomous AI Research Agents Across Quality, Diversity and Novelty](https://arxiv.org/abs/2606.25198v2). Closely related autonomous research evaluation, including LM pretraining.
7. Mostafa Dehghani et al. (2018; ICLR 2019). [Universal Transformers](https://arxiv.org/abs/1807.03819v3). Established recurrent computation and adaptive halting.
8. Jonas Geiping et al. (2025). [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171v2). Established recurrent-depth language modeling; compute differs from unique parameter count.
9. Ronen Eldan and Yuanzhi Li (2023). [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759v2). Optional small-model language domain; task sensitivity must be checked.
10. Michael Poli et al. (2024). [Mechanistic Design and Scaling of Hybrid Architectures](https://arxiv.org/abs/2403.17844v2). Architecture screening with mechanistic synthetic tasks.
11. Simran Arora et al. (2023). [Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927). Recall probes and gaps between simple synthetic tests and language behavior.
12. George E. Dahl et al. (2023; revised 2025). [Benchmarking Neural Network Training Algorithms](https://arxiv.org/abs/2306.07179). AlgoPerf; tuning and cost-comparison discipline.
13. Cynthia Dwork et al. (2015). [Generalization in Adaptive Data Analysis and Holdout Reuse](https://arxiv.org/abs/1506.02629). Protecting confirmation from adaptive selection.
14. Chenglei Si et al. (2026). [Towards Execution-Grounded Automated AI Research](https://arxiv.org/abs/2601.14525v1). Executable evidence and research-agent evaluation.
15. Chenglei Si, Tatsunori Hashimoto, and Diyi Yang (2025). [The Ideation-Execution Gap: Execution Outcomes of LLM-Generated versus Human Research Ideas](https://arxiv.org/abs/2506.20803v1). Distinguishing proposal ratings from implemented outcomes.
16. Lianmin Zheng et al. (2023). [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685). Judge biases; not a validated scientific-creativity rubric.
17. Yuto Suzuki and Farnoush Banaei-Kashani (2025; v3, 2026). [Universe of Thoughts: A Computational Framework for Creative Reasoning in Large Language Models](https://arxiv.org/abs/2511.20471v3). Mutable assumptions and task-preserving transformations.
18. Qing Zhao et al. (2026). [EvoSCM: Scientific Belief Revision Through Causal Model Evolution and Experimentation](https://arxiv.org/abs/2609.01526v2). Evidence-driven structural revision; causal models differ from our justification DAG.
19. Hugging Face FineData team. [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). Candidate source for a frozen English web-text subset.
20. Jared Kaplan et al. (2020). [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361). Approximate nonembedding Transformer training-compute accounting; not a measured runtime guarantee.

---

## Implementation choices in this repository

The spec leaves these open; the code currently uses the following. All are settings, not
commitments, and the Stage A defaults are meant to be revisited before Stage B is frozen.

- **Corpus and model size:** TinyStories (the spec's debugging option), BPE vocabulary 4,096,
  context 512, batch 32. The baseline is a 5.9M-parameter Llama-style decoder (d256, 6 layers),
  smaller than the spec's 10–30M because of the available GPU. FineWeb-Edu preparation is
  supported but has not been profiled.
- **Smoke checkpoints:** b1 = 0.05 and b2 = 0.1 of a 5.5e15-FLOP horizon (about 20 tokens per
  parameter for the baseline). FLOPs are counted as executed, identically on CPU and GPU.
- **Mechanism probes:** in-context copying at several delays, and dev loss by context position.
- **Screening rule (provisional):** advance if delta at b2 <= 0; reject if delta at b2 >= 0.05
  and the gap is not closing from b1; otherwise hold. **Usefulness:** delta at the full
  horizon <= -0.02 nats; within 0.01 of that threshold is uncertain.
- **Calibration cohort:** cap 10 agent candidates, half of the slots reserved for
  reformulations (the agent's pre-result label or a researcher override), plus three known
  controls (recurrent depth, depth-weighted averaging, wide-shallow). The long runs use fresh
  seeds.
- **Agents:** OpenAI chat models with function tools (gpt-5.4-mini for testing). Each job
  allows up to 16 model calls and 2 implementation repairs.
- **Confirmation (section E2):** paired seeds; Bonferroni-adjusted t-intervals across up to
  three candidates; the confirm split stays sealed outside campaign archives.
