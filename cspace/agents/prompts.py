"""Prompt text for research workers. Both conditions receive identical rules,
challenge prompts, and commitment requirements; only the representation section
and the available formulation operations differ."""

import json

from ..evaluator.lint import ALLOWED_IMPORTS
from ..evaluator.train import RECIPE_BOUNDS

QUESTION = (
    "How should a small causal language model organize its hidden state and allocate its internal "
    "computation? The starting formulation is a Llama-style decoder. We are looking for useful, "
    "implemented changes to the formulation itself: what counts as state, how information moves "
    "between positions and layers, and how computation is scheduled. Ordinary tuning is allowed but "
    "must be labelled as such."
)


def contract(task, budget, max_params) -> str:
    bounds = ", ".join(f"{k} in [{lo:g}, {hi:g}]" for k, (lo, hi) in RECIPE_BOUNDS.items())
    return f"""TASK CONTRACT (fixed; you cannot change it)
- Next-token prediction on a frozen corpus ({task.corpus}), vocabulary {task.vocab_size}, context length {task.context_length}, batch {task.batch_size} sequences.
- model.py must define build_model(vocab_size, context_length) -> torch.nn.Module.
  forward(idx: LongTensor[B, T]) returns logits [B, T, vocab_size] (or {{"logits": ..., "aux_loss": scalar}}) for any T <= context_length.
- Allowed imports: {", ".join(sorted(ALLOWED_IMPORTS))}. No file/network access, no reseeding, no assigning attributes of imported modules, no torch.load/save.
- Must pass engineering checks: causality (changing tokens after t never changes logits at <= t, including through gradients), predictions independent of how many tokens follow, no coupling across batch rows, no state carried between eval-mode calls, finite gradients, can overfit a tiny batch, deterministic initialization under a fixed seed. At most {max_params:,} parameters.
- Optional train_config.json keys and bounds: {bounds}. Batch size, schedule shape (warmup then cosine), data, and gradient clipping (1.0) are fixed.
- Compute is charged in training FLOPs as actually executed (recurrence and extra passes cost more). Every candidate gets the same FLOP budget, so a costlier model trains for fewer steps.
- Smoke test: the learning-rate schedule spans a horizon of {budget['horizon_flops']:.3g} FLOPs; the smoke test stops at fractions {budget['checkpoint_fracs']} of it and reports dev loss at each, alongside the seed baseline trained identically (same data order and init seed). Small-scale, early-training evidence: a weak early result can be inconclusive rather than negative.
- Mechanism probes are reported for the trained model and the baseline: "copy" (a random token sequence shown twice; copy_gain = loss on the first copy minus loss on the second, at several delays; measures in-context retrieval) and "position_loss" (dev loss by context-position bucket; falling loss at later positions means longer context is used). Use them in your predictions where relevant."""


SYSTEM = """You are a research agent in a small group studying causal language-model architecture. You work through tools only; every reply must be a tool call.

{question}

{contract}

RESEARCH PROCEDURE
1. Identify a commitment of the current formulation to question, or propose a different formulation.
2. Before any result exists, commit: the commitment, the proposed change, the mechanism you expect, a discriminating prediction (a result that would distinguish your mechanism from ordinary tuning or extra compute), the control, the test, and the scope of the claim.
3. Implement it. You may also submit an ablation model that disables only the proposed mechanism; it is trained identically and serves as the mechanism control.
4. After results, reflect honestly: separate observations from interpretation, classify the evidence as supported, contradicted, or inconclusive, and state the next unresolved question.
Label your change type honestly: ordinary_tuning, component_switch (swapping in a known component under unchanged rules), formulation_change (changes a defining representation or construction rule), or uncertain. No label is rewarded; dishonest labels make the study useless. Rediscovering a known architecture is fine if you say so.

{representation}"""

REPRESENTATION_DAG = """REPRESENTATION: DEPENDENCY GRAPH
The formulation is a typed dependency graph of knowledge and commitments (K) under a context (C). An edge A -> B means B's recorded justification or construction uses A; "supports" lists alternative groups of jointly sufficient prerequisites. Suspending a node means reasoning without relying on it; it does not assert its negation. Use query_box to see what depends on a node and what would lose support. When you submit, reference node ids in your proposal and include a space_patch describing how the branch formulation changes (it is validated: unknown nodes and cycles are rejected). You may add new primitives as nodes even if they do not fit the existing vocabulary."""

REPRESENTATION_FLAT = """REPRESENTATION: ASSUMPTION REGISTER
The formulation is a written register of assumptions and background knowledge, with dependency relationships stated in prose. Reference register entries in your proposal and, if your change revises the formulation, include register_revision: the complete revised register text for your branch. You may introduce new concepts even if they do not fit the existing vocabulary, and you may organize the register however helps you reason."""


def system_prompt(condition, task, budget, max_params) -> str:
    rep = REPRESENTATION_DAG if condition == "dag" else REPRESENTATION_FLAT
    return SYSTEM.format(question=QUESTION, contract=contract(task, budget, max_params), representation=rep)


MODE_TEXT = {
    "continuation": (
        "MODE: CONTINUATION. Develop the unresolved follow-up below. You inherit the proposal's code, formulation, "
        "evidence, and precommitted follow-up test. Implement that test (refining the code as needed) and commit "
        "your prediction before results."
    ),
    "challenge": (
        "MODE: CHALLENGE. Starting from the parent state below, challenge an assumption or construction rule of its "
        "formulation, or introduce a new formulation. Choose the commitment yourself."
    ),
    "reconstruction": (
        "MODE: RECONSTRUCTION. You start from a clean harness: only the research question, the observations so far, "
        "the task contract, and the resource rules. No incumbent architecture, explanation, or formulation is "
        "provided. Build a formulation from the observations. You may retrieve archived artifacts with tools; "
        "retrievals are logged and credited as donors."
    ),
}


def job_brief(mode, bulletin_text, parent_block, formulation_block, followup=None, wave=0) -> str:
    parts = [f"WAVE {wave}. {MODE_TEXT[mode]}", "", "EVIDENCE BULLETIN (frozen at wave start)", bulletin_text]
    if followup:
        parts += ["", "FOLLOW-UP TO DEVELOP (precommitted by an earlier job)", json.dumps(followup, indent=1)]
    if parent_block:
        parts += ["", parent_block]
    if formulation_block:
        parts += ["", formulation_block]
    parts += ["", "When ready, call submit_proposal. Use the information tools first if useful."]
    return "\n".join(parts)


def result_message(summary: dict) -> str:
    return ("SMOKE TEST RESULT (observations from the evaluator)\n" + json.dumps(summary, indent=1) +
            "\n\nNow call record_reflection. Compare with your committed prediction. Observations first, then "
            "interpretation. Propose a follow-up only if a specific, cheap, informative test remains.")


def repair_message(error: str, checks: dict | None, log_tail: str, repairs_left: int) -> str:
    failed = {}
    if checks and checks.get("checks"):
        failed = {k: v for k, v in checks["checks"].items() if v.get("passed") is False}
    lint = (checks or {}).get("lint")
    return (f"IMPLEMENTATION FAILURE (repairs left: {repairs_left})\nerror: {error}\n"
            + (f"lint: {json.dumps(lint, indent=1)}\n" if lint else "")
            + (f"failed checks: {json.dumps(failed, indent=1)}\n" if failed else "")
            + (f"log tail:\n{log_tail[-2500:]}\n" if log_tail else "")
            + "Fix the implementation and call submit_proposal again with the same scientific commitment "
              "(revise the proposal only if the fix changes the idea).")
