"""Command-line interface: python -m cspace <command> ..."""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must precede CUDA init

from .paths import DATA_DIR, SEEDS_DIR  # noqa: E402

BASELINE_DIR = SEEDS_DIR / "baseline_decoder"


def _default_task_dir():
    from .task import default_task_dir

    try:
        return default_task_dir()
    except ValueError as e:
        sys.exit(f"{e}; pass --task")


def _executor_spec(args):
    """Executor spec from --executor (kind or JSON) and --slots; None keeps the default."""
    if not args.executor and not args.slots:
        return None
    spec = (json.loads(args.executor) if args.executor and args.executor.startswith("{")
            else {"kind": args.executor or "local"})
    if args.slots:
        spec["slots"] = args.slots
    from .executor import normalize_spec

    return normalize_spec(spec)


def _hardware(args):
    """device / throttle / temp_limit_c / executor settings given on the command line."""
    return dict(device=args.device, throttle=args.throttle, temp_limit_c=args.temp_limit,
                executor=_executor_spec(args))


def _fracs(text):
    return tuple(float(f) for f in text.split(",") if f.strip())


# ---------------------------------------------------------------------------
def cmd_prepare_data(args):
    from .data.prepare import prepare

    out = Path(args.out) if args.out else DATA_DIR / args.corpus
    prepare(args.corpus, out, train_docs=args.train_docs, heldout_docs=args.heldout_docs,
            vocab_size=args.vocab, context_length=args.context, batch_size=args.batch,
            tokenizer_docs=args.tokenizer_docs)


def _seconds_per_step(model, opt, x, y, ctx, steps, device, warmup=5):
    """Mean wall-clock seconds of an optimizer step after `warmup` untimed steps."""
    import torch

    def step():
        with ctx():
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        step()
    sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    sync()
    return (time.perf_counter() - t0) / steps


def cmd_profile(args):
    import torch

    from .evaluator.candidate import load_candidate_module
    from .evaluator.determinism import configure
    from .evaluator.metrics import count_params, measure_step_flops
    from .evaluator.train import autocast_for

    configure(deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = autocast_for(device)
    module = load_candidate_module(BASELINE_DIR / "model.py")
    sizes = [dict(d_model=256, n_layers=6, n_heads=4), dict(d_model=384, n_layers=8, n_heads=6),
             dict(d_model=512, n_layers=8, n_heads=8), dict(d_model=512, n_layers=12, n_heads=8)]
    V, T, B = args.vocab, args.context, args.batch
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"device {name}; vocab {V}, context {T}, batch {B} ({B * T:,} tokens/step); bf16 autocast")
    print(f"{'config':<18}{'total':>9}{'non-emb':>9}{'GFLOP/step':>12}{'tok/s':>10}{'TFLOP/s':>9}"
          f"{'peak GB':>9}{'min @20 tok/param':>19}")
    for cfg in sizes:
        module.CONFIG = {**module.CONFIG, **cfg}
        torch.manual_seed(0)
        model = module.build_model(V, T).to(device)
        params = count_params(model)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=device.type == "cuda")
        x = torch.randint(0, V, (B, T), device=device)
        y = torch.randint(0, V, (B, T), device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        flops = measure_step_flops(model, x, y, ctx)

        dt = _seconds_per_step(model, opt, x, y, ctx, args.steps, device)
        tok_s = B * T / dt
        tflops = flops / dt / 1e12
        minutes = 20 * params["total"] / tok_s / 60
        peak = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else float("nan")
        label = f"d{cfg['d_model']} L{cfg['n_layers']} h{cfg['n_heads']}"
        print(f"{label:<18}{params['total'] / 1e6:>8.1f}M{params['nonembedding'] / 1e6:>8.1f}M"
              f"{flops / 1e9:>12.1f}{tok_s:>10,.0f}{tflops:>9.1f}{peak:>9.2f}{minutes:>19.1f}")
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()


def cmd_seed_archive(args):
    from .archive import Archive

    with Archive() as archive:
        record, created = archive.add_candidate(
            BASELINE_DIR, action_mode="seed",
            notes="Llama-style starting decoder (RoPE, RMSNorm, SwiGLU, tied embeddings).")
    print(f"{'added' if created else 'already present'}: {record['id']}")


def cmd_check(args):
    from .evaluator.candidate import load_candidate_module
    from .evaluator.checks import run_checks
    from .evaluator.determinism import configure
    from .evaluator.evaluate import resolve_device
    from .evaluator.lint import lint_file
    from .task import load_task

    configure()
    bundle = Path(args.bundle)
    violations = lint_file(bundle / "model.py")
    if violations:
        print("lint violations:\n  " + "\n  ".join(violations))
        sys.exit(1)
    task = load_task(args.task or _default_task_dir())
    module = load_candidate_module(bundle / "model.py")
    report = run_checks(module.build_model, task.vocab_size, task.context_length, resolve_device(args.device))
    for name, item in report["checks"].items():
        mark = {True: "PASS", False: "FAIL", None: "skip"}[item["passed"]]
        req = "" if item["required"] else " (diagnostic)"
        print(f"  {mark}  {name}{req}  {item['detail']}")
    print("valid" if report["passed"] else "INVALID")
    sys.exit(0 if report["passed"] else 1)


def _run_evaluation(bundle_dir, task_dir, split, budget, seeds, args):
    request = dict(bundle_dir=str(bundle_dir), task_dir=str(task_dir), split=split, budget=budget,
                   seeds=seeds, device=args.device, allow_sealed=getattr(args, "allow_sealed", False),
                   time_limit_s=args.time_limit, throttle=args.throttle, temp_limit_c=args.temp_limit)
    if args.no_isolate:
        from .evaluator.evaluate import evaluate_bundle

        return evaluate_bundle(**request)
    from .executor import get_executor

    return get_executor(_executor_spec(args) or {"kind": "local", "stream": True}).run(request)


def _summarize(result):
    print(f"status: {result['status']}" + (f"  ({result['error'].splitlines()[0]})" if result.get("error") else ""))
    for ck in result.get("checkpoints") or []:
        print(f"  frac {ck['frac']:<7.4g} step {ck['step']:>6}  {ck['flops']:.3g} FLOPs  "
              f"dev loss {ck['dev_loss']:.4f}  {ck['train_seconds']:.0f}s")
    if result.get("final"):
        print(f"  final {result['final']['split']} loss {result['final']['loss']:.4f}")


def cmd_evaluate(args):
    from .archive import Archive

    task_dir = args.task or _default_task_dir()
    budget = dict(horizon_flops=args.horizon_flops, checkpoint_fracs=list(_fracs(args.checkpoints)),
                  eval_tokens=args.eval_tokens)
    seeds = {"init": args.seed_init, "data": args.seed_data}
    with Archive() as archive:
        if args.candidate:
            candidate = archive.get_candidate(args.candidate)
            if candidate is None:
                sys.exit(f"unknown candidate {args.candidate}")
        else:
            candidate, _ = archive.add_candidate(args.bundle, action_mode="manual", notes=args.notes or "")
        with tempfile.TemporaryDirectory(prefix="cspace_bundle_") as tmp:
            bundle_dir = archive.materialize(candidate["bundle_hash"], tmp)
            result = _run_evaluation(bundle_dir, task_dir, args.split, budget, seeds, args)
        record = archive.add_experiment(result, candidate["id"], purpose=args.purpose or "")
    print(f"candidate {candidate['id']}  experiment {record['id']}")
    _summarize(result)


def cmd_replay(args):
    from .archive import Archive

    with Archive() as archive:
        original = archive.get_experiment(args.experiment)
        if original is None:
            sys.exit(f"unknown experiment {args.experiment}")
        if original["status"] != "ok":
            sys.exit(f"experiment status is {original['status']}; only completed runs can be replayed")
        candidate = archive.get_candidate(original["candidate_id"])
        task_dir = args.task or _default_task_dir()
        with tempfile.TemporaryDirectory(prefix="cspace_bundle_") as tmp:
            bundle_dir = archive.materialize(candidate["bundle_hash"], tmp)
            args.allow_sealed = original["split"] == "confirm"
            result = _run_evaluation(bundle_dir, task_dir, original["split"], original["budget"],
                                     original["seeds"], args)
        record = archive.add_experiment(result, candidate["id"], replay_of=original["id"], purpose="replay")
    print(f"replay {record['id']} of {original['id']}")
    problems = []
    for key in ("evaluator_version", "task_hash", "bundle_hash", "attention_backend"):
        if result.get(key) != original.get(key):
            problems.append(f"{key} differs: {original.get(key)} -> {result.get(key)}")
    if result["status"] != "ok":
        problems.append(f"replay status {result['status']}")
    else:
        for a, b in zip(original["checkpoints"], result["checkpoints"]):
            diff = abs(a["dev_loss"] - b["dev_loss"])
            flag = "ok" if diff <= args.tolerance else "MISMATCH"
            print(f"  frac {a['frac']:<7.4g} original {a['dev_loss']:.5f}  replay {b['dev_loss']:.5f}  "
                  f"|diff| {diff:.2e}  {flag}")
            if diff > args.tolerance:
                problems.append(f"checkpoint {a['frac']} differs by {diff:.2e}")
    if problems:
        print("REPLAY FAILED:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print(f"replay reproduced all checkpoints within {args.tolerance:g}")


def cmd_show(args):
    from .archive import Archive

    with Archive() as archive:
        for c in archive.list_candidates():
            exps = archive.list_experiments(c["id"])
            print(f"{c['id']}  {c['action_mode']:<14} parent={c['research_parent_id']}  {c['notes'][:60]}")
            for e in exps:
                final = f"{e['final']['split']} {e['final']['loss']:.4f}" if e.get("final") else "-"
                fl = (e.get("costs") or {}).get("train_flops")
                print(f"    {e['id']}  {e['status']:<8} {e['split']:<7} final={final:<14} "
                      f"flops={fl or 0:.3g}  replay_of={e.get('replay_of')}")


def cmd_ledger(args):
    from .archive import Archive

    with Archive() as archive:
        for row in archive.ledger_summary():
            print(f"{row['ledger']:<15}{row['unit']:<16}{row['total']:>16.4g}  ({row['n']} entries)")


def _llm(model, effort):
    from .agents.llm import OpenAIChat

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set in the environment")
    return OpenAIChat(model, reasoning_effort=effort)


def cmd_llm_ping(args):
    llm = _llm(args.model, args.effort)
    tool = {"type": "function", "function": {"name": "answer", "description": "Return the answer.",
                                             "parameters": {"type": "object", "required": ["text"],
                                                            "properties": {"text": {"type": "string"}}}}}
    reply = llm.chat([{"role": "user", "content": "Call the answer tool with the text 'ok'."}], [tool])
    print(f"model {reply.usage['model']}: tool call {[(c.name, c.arguments) for c in reply.tool_calls]}; usage {reply.usage}")


def cmd_campaign_init(args):
    from .agents.campaign import Campaign

    with Campaign.create(args.name, condition=args.condition, model=args.model, reasoning_effort=args.effort,
                         box=args.box, task=args.task, horizon_flops=args.horizon_flops,
                         smoke_fracs=list(_fracs(args.smoke_fracs)) if args.smoke_fracs else None,
                         max_turns=args.max_turns, parallel_jobs=args.parallel_jobs, **_hardware(args)) as camp:
        print(f"created campaign {args.name} in {camp.dir}")
        print(json.dumps(camp.config, indent=2))


def cmd_campaign_run(args):
    from .agents.campaign import Campaign

    with Campaign(args.name) as camp:
        camp.run(args.waves, _llm(camp.config["model"], camp.config.get("reasoning_effort")))


def cmd_campaign_status(args):
    from .agents.campaign import Campaign

    with Campaign(args.name) as camp:
        cfg = camp.config
        print(f"campaign {args.name}: condition {cfg['condition']}, model {cfg['model']}, "
              f"next wave {camp._meta_get('next_wave')}")
        ctl = camp._meta_get("control_experiment_id")
        if ctl:
            c = camp.archive.get_experiment(ctl)
            print("baseline control: " + ", ".join(f"@{k['frac']:g} {k['dev_loss']:.4f}" for k in c["checkpoints"]))
        for job in camp.jobs():
            p = job["plan"]
            cks = (job.get("summary") or {}).get("checkpoints") or []
            delta = f"delta {cks[-1]['delta_vs_baseline']:+.4f}" if cks else "-"
            evidence = (job.get("reflection") or {}).get("evidence_class") or "-"
            print(f"  {p['id']} {p['mode']:<14} parent={p['research_parent_id'] or '-':<18} {job['status']:<22} "
                  f"{job.get('final_candidate_id') or '-':<18} {job.get('change_type') or '-':<18} {delta:<14} "
                  f"{evidence}")
        if camp.valid_states():
            print(f"incumbent: {camp.incumbent()}")
        for row in camp.archive.ledger_summary():
            print(f"  ledger {row['ledger']:<14}{row['unit']:<20}{row['total']:>16,.6g}")


def cmd_campaign_job(args):
    from .agents.campaign import Campaign

    with Campaign(args.name) as camp:
        job = next((j for j in camp.jobs() if j["plan"]["id"] == args.job), None)
        if job is None:
            sys.exit(f"unknown job {args.job}")
        if args.transcript and job.get("transcript"):
            for m in json.loads(camp.archive.get_bytes(job["transcript"])):
                body = m.get("content") or json.dumps(m.get("tool_calls"), indent=1)
                print(f"--- {m['role']}\n{body}\n")
        else:
            print(json.dumps(job, indent=1))


def cmd_cohort_init(args):
    from .calibration import Cohort

    overrides = dict(sources=args.source or [], cap=args.cap, reform_quota=args.reform_quota, task=args.task,
                     horizon_flops=args.horizon_flops, **_hardware(args),
                     smoke_fracs=list(_fracs(args.smoke_fracs)) if args.smoke_fracs else None,
                     long_fracs=list(_fracs(args.long_fracs)) if args.long_fracs else None)
    if args.controls is not None:
        overrides["controls"] = [c for c in args.controls.split(",") if c]
    if args.overrides:
        overrides["stratum_overrides"] = json.loads(Path(args.overrides).read_text(encoding="utf-8"))
    with Cohort.create(args.name, **overrides) as coh:
        print(f"created cohort {args.name} in {coh.dir} (spec frozen, hash {coh.spec_hash[:12]})")
        print(json.dumps(coh.spec, indent=2))


def cmd_cohort(args):
    from .calibration import Cohort

    with Cohort(args.name) as coh:
        if args.cohort_command == "status":
            print(coh.status())
        elif args.cohort_command == "report":
            coh.report()
            print((coh.dir / "report.md").read_text(encoding="utf-8"))
        else:  # "run" goes up to --until; a phase name runs up to that phase
            coh.run(until=args.until if args.cohort_command == "run" else args.cohort_command)


def cmd_campaign_report(args):
    from .reporting import report_campaign

    rep = report_campaign(args.name, delta=args.delta)
    print((rep["dir"] / "summary.md").read_text(encoding="utf-8"))
    print(f"wrote ledger, jobs, case cards and summary to {rep['dir']}")


def cmd_study_compare(args):
    from .reporting import compare

    pairs = []
    for p in args.pair:
        dag, _, flat = p.partition(":")
        if not dag or not flat:
            sys.exit(f"--pair expects DAG_CAMPAIGN:FLAT_CAMPAIGN, got {p!r}")
        pairs.append((dag, flat))
    compare(args.study, pairs, delta=args.delta, seed=args.seed, allow_mismatch=args.allow_mismatch)


def cmd_confirm(args):
    from .confirm import Confirmation

    if args.confirm_command == "init":
        with Confirmation.create(
                args.name, campaign=args.campaign, candidates=args.candidate or None, split=args.split,
                seeds=[int(x) for x in args.seeds.split(",")] if args.seeds else None,
                fracs=list(_fracs(args.fracs)) if args.fracs else None, threshold=args.threshold,
                **_hardware(args)) as conf:
            print(f"created confirmation {args.name} (spec {conf.spec_hash[:12]}): candidates "
                  f"{conf.spec['candidates']}, baseline {conf.spec['baseline']}, split {conf.spec['split']}, "
                  f"seeds {conf.spec['seeds']}")
        return
    with Confirmation(args.name) as conf:
        conf.run() if args.confirm_command == "run" else conf.report()
        print((conf.dir / "report.md").read_text(encoding="utf-8"))


def cmd_box(args):
    from .box import load_space, render_markdown, validate_space
    from .agents.space_ops import query, render_dag, render_flat

    space = load_space(args.path)
    validate_space(space)
    if args.box_command == "validate":
        print(f"valid: {len(space['nodes'])} nodes")
    elif args.box_command == "render":
        text = {"md": render_markdown, "dag": render_dag, "flat": render_flat}[args.format](space)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(text)
    elif args.box_command == "query":
        print(json.dumps(query(space, args.node), indent=1))


def cmd_freeze_evaluator(args):
    from .evaluator.version import LOCK_PATH, freeze

    lock = freeze()
    print(f"evaluator {lock['version']} locked in {LOCK_PATH}")


# ---------------------------------------------------------------------------
def main(argv=None):
    def add_hardware_args(s, device=None, throttle=None):
        s.add_argument("--device", choices=["auto", "cuda", "cpu"], default=device)
        s.add_argument("--throttle", type=float, default=throttle,
                       help="fraction of wall-clock the GPU idles between steps, e.g. 0.5 (results unchanged)")
        s.add_argument("--temp-limit", type=int, help="pause training while the GPU is at or above this temperature (C)")
        s.add_argument("--executor", help="executor kind or JSON spec, e.g. local, loopback, "
                                          "'{\"kind\": \"modal\", \"slots\": 4}'")
        s.add_argument("--slots", type=int, help="concurrent evaluations through the executor")

    p = argparse.ArgumentParser(prog="cspace", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("prepare-data", help="download a corpus subset, train a tokenizer, write a task")
    s.add_argument("--corpus", default="tinystories", choices=["tinystories", "fineweb-edu"])
    s.add_argument("--out")
    s.add_argument("--train-docs", type=int, default=200_000)
    s.add_argument("--heldout-docs", type=int, default=20_000)
    s.add_argument("--tokenizer-docs", type=int, default=50_000)
    s.add_argument("--vocab", type=int, default=4096)
    s.add_argument("--context", type=int, default=512)
    s.add_argument("--batch", type=int, default=32)
    s.set_defaults(func=cmd_prepare_data)

    s = sub.add_parser("profile", help="measure baseline throughput at several sizes (synthetic tokens)")
    s.add_argument("--vocab", type=int, default=4096)
    s.add_argument("--context", type=int, default=512)
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--steps", type=int, default=20)
    s.set_defaults(func=cmd_profile)

    s = sub.add_parser("seed-archive", help="add the baseline decoder to the archive")
    s.set_defaults(func=cmd_seed_archive)

    s = sub.add_parser("check", help="lint and run engineering checks on a bundle directory")
    s.add_argument("bundle")
    s.add_argument("--task")
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_check)

    def add_run_args(s):
        s.add_argument("--task")
        s.add_argument("--time-limit", type=float, default=3600.0, help="graceful limit in seconds")
        s.add_argument("--no-isolate", action="store_true", help="run in-process (debugging)")
        add_hardware_args(s, device="auto", throttle=0.0)

    s = sub.add_parser("evaluate", help="evaluate an archived candidate or a bundle directory")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--candidate")
    g.add_argument("--bundle")
    s.add_argument("--split", default="dev", choices=["dev", "confirm"])
    s.add_argument("--allow-sealed", action="store_true", help="required for the confirm split")
    s.add_argument("--horizon-flops", type=float, default=2e15)
    s.add_argument("--checkpoints", default="1.0", help="comma-separated fractions of the horizon")
    s.add_argument("--eval-tokens", type=int, default=1 << 20)
    s.add_argument("--seed-init", type=int, default=0)
    s.add_argument("--seed-data", type=int, default=0)
    s.add_argument("--purpose")
    s.add_argument("--notes")
    add_run_args(s)
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("replay", help="re-run an archived experiment and compare measurements")
    s.add_argument("experiment")
    s.add_argument("--tolerance", type=float, default=2e-3)
    add_run_args(s)
    s.set_defaults(func=cmd_replay)

    sub.add_parser("show", help="list candidates and experiments").set_defaults(func=cmd_show)
    sub.add_parser("ledger", help="summarize cost ledgers").set_defaults(func=cmd_ledger)
    sub.add_parser("freeze-evaluator", help="lock the evaluator source hash").set_defaults(func=cmd_freeze_evaluator)

    s = sub.add_parser("box", help="validate, render, or query a box (conceptual space) file")
    bsub = s.add_subparsers(dest="box_command", required=True)
    b = bsub.add_parser("validate")
    b.add_argument("path")
    b.set_defaults(func=cmd_box)
    b = bsub.add_parser("render")
    b.add_argument("path")
    b.add_argument("--format", choices=["md", "dag", "flat"], default="md",
                   help="md: dossier; dag/flat: exactly what each condition's agents see")
    b.add_argument("--out")
    b.set_defaults(func=cmd_box)
    b = bsub.add_parser("query")
    b.add_argument("path")
    b.add_argument("node")
    b.set_defaults(func=cmd_box)

    s = sub.add_parser("llm-ping", help="check the OpenAI key and a model with one tool call")
    s.add_argument("--model", default="gpt-5.4-mini")
    s.add_argument("--effort")
    s.set_defaults(func=cmd_llm_ping)

    s = sub.add_parser("campaign", help="research campaigns")
    csub = s.add_subparsers(dest="campaign_command", required=True)
    c = csub.add_parser("init", help="create a campaign (own archive, seeded with the baseline)")
    c.add_argument("name")
    c.add_argument("--condition", choices=["dag", "flat"], default="dag")
    c.add_argument("--model")
    c.add_argument("--effort", help="reasoning effort for reasoning models")
    c.add_argument("--box")
    c.add_argument("--task")
    c.add_argument("--horizon-flops", type=float)
    c.add_argument("--smoke-fracs", help="comma-separated fractions of the horizon")
    c.add_argument("--max-turns", type=int)
    add_hardware_args(c)
    c.add_argument("--parallel-jobs", type=int, help="jobs of a wave run concurrently (default 1)")
    c.set_defaults(func=cmd_campaign_init)
    c = csub.add_parser("run", help="run waves (resumes where it stopped)")
    c.add_argument("name")
    c.add_argument("--waves", type=int, default=1)
    c.set_defaults(func=cmd_campaign_run)
    c = csub.add_parser("status", help="jobs, incumbent, and cost ledgers")
    c.add_argument("name")
    c.set_defaults(func=cmd_campaign_status)
    c = csub.add_parser("job", help="show a job record or its transcript")
    c.add_argument("name")
    c.add_argument("job")
    c.add_argument("--transcript", action="store_true")
    c.set_defaults(func=cmd_campaign_job)
    c = csub.add_parser("report", help="case ledger, case cards (full and condition-neutral), summary")
    c.add_argument("name")
    c.add_argument("--delta", type=float, default=0.02, help="improvement threshold in nats for counting")
    c.set_defaults(func=cmd_campaign_report)

    s = sub.add_parser("study", help="Stage B: compare matched DAG/flat campaign pairs")
    ssub = s.add_subparsers(dest="study_command", required=True)
    c = ssub.add_parser("compare", help="matched-pair comparison plus a blinded case pack")
    c.add_argument("study")
    c.add_argument("--pair", action="append", required=True, help="DAG_CAMPAIGN:FLAT_CAMPAIGN (repeatable)")
    c.add_argument("--delta", type=float, default=0.02)
    c.add_argument("--seed", type=int, default=0, help="shuffle seed for blinded cases")
    c.add_argument("--allow-mismatch", action="store_true", help="report, rather than refuse, unmatched settings")
    c.set_defaults(func=cmd_study_compare)

    s = sub.add_parser("confirm", help="seed replication (dev) or sealed confirmation (confirm split)")
    fsub = s.add_subparsers(dest="confirm_command", required=True)
    c = fsub.add_parser("init", help="freeze candidates, seeds and split")
    c.add_argument("name")
    c.add_argument("--campaign", required=True)
    c.add_argument("--candidate", action="append", help="candidate id or 'incumbent' (repeatable; default incumbent)")
    c.add_argument("--split", choices=["dev", "confirm"])
    c.add_argument("--seeds", help="comma-separated seeds, e.g. 1,2,3")
    c.add_argument("--fracs", help="checkpoints as fractions of the campaign horizon (last is analysed)")
    c.add_argument("--threshold", type=float, help="practical improvement threshold (nats)")
    add_hardware_args(c)
    c.set_defaults(func=cmd_confirm)
    for name in ("run", "report"):
        c = fsub.add_parser(name, help="run missing evaluations and report" if name == "run" else "report only")
        c.add_argument("name")
        c.set_defaults(func=cmd_confirm)

    s = sub.add_parser("cohort", help="Stage A calibration cohort: smoke tests vs longer runs")
    hsub = s.add_subparsers(dest="cohort_command", required=True)
    c = hsub.add_parser("init", help="create a cohort with a frozen spec")
    c.add_argument("name")
    c.add_argument("--source", action="append", help="source campaign name (repeatable)")
    c.add_argument("--controls", help="comma-separated control names under seeds/controls or bundle paths ('' for none)")
    c.add_argument("--cap", type=int, help="agent candidates to enroll (default 10)")
    c.add_argument("--reform-quota", type=float, help="share of slots reserved for reformulations (default 0.5)")
    c.add_argument("--overrides", help="JSON file mapping source candidate id -> reformulation|ordinary")
    c.add_argument("--task")
    c.add_argument("--horizon-flops", type=float)
    c.add_argument("--smoke-fracs", help="b1,b2")
    c.add_argument("--long-fracs", help="checkpoints of the long run, ending at 1")
    add_hardware_args(c)
    c.set_defaults(func=cmd_cohort_init)
    for name, text in (("enroll", "enroll members by the frozen outcome-blind rule"),
                       ("smoke", "run smoke tests for every member and the baseline"),
                       ("screen", "freeze advance/hold/reject decisions"),
                       ("long", "run long runs for every member (requires frozen screening)"),
                       ("report", "write report.md/json, members.csv, and the plot"),
                       ("status", "phase and member status")):
        c = hsub.add_parser(name, help=text)
        c.add_argument("name")
        c.set_defaults(func=cmd_cohort)
    c = hsub.add_parser("run", help="run all phases up to --until (resumes)")
    c.add_argument("name")
    c.add_argument("--until", default="report", choices=["enroll", "smoke", "screen", "long", "report"])
    c.set_defaults(func=cmd_cohort)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
