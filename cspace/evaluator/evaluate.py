"""evaluate(candidate, split, budget): the single entry point for measurements.

Pipeline: lint -> load -> engineering checks -> compute-capped training with
dev checkpoints -> final evaluation on the requested split. The confirm split
is sealed unless allow_sealed=True (never passed by agent-facing tooling).

Status values:
  ok        trained to the final checkpoint
  invalid   candidate violates the contract (lint, bad train_config, failed checks)
  diverged  non-finite training loss
  timeout   time limit reached before the final checkpoint
  oom       CUDA out of memory
  overheat  GPU temperature guard stopped the run (see pacing)
  refused   request not permitted (e.g. sealed split)
  error     candidate code raised during training/evaluation, or harness failure

Pacing (throttle, temp_limit_c) changes wall-clock time only, never measurements.
Mechanism probes (cspace.evaluator.probes) run after a completed run unless probes=False.
"""

import json
import time
import traceback
from pathlib import Path

import torch

from ..hashing import bundle_hash, bundle_manifest
from ..task import load_task
from .candidate import CandidateError, load_candidate_module
from .checks import run_checks
from .determinism import configure
from .lint import lint_file
from .metrics import count_params
from .pacing import Pacer
from .probes import run_probes
from .train import Budget, attention_context, autocast_for, eval_loss, resolve_recipe, train_candidate
from ..data.loader import TokenStream
from .version import check_lock, evaluator_version

DEFAULT_MAX_PARAMS = 60_000_000


def resolve_device(device="auto"):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def evaluate_bundle(bundle_dir, task_dir, split="dev", budget=None, seeds=None, device="auto",
                    allow_sealed=False, time_limit_s=None, max_params=DEFAULT_MAX_PARAMS,
                    enforce_lock=True, throttle=0.0, temp_limit_c=None, probes=True, log=print) -> dict:
    configure(deterministic=True)
    bundle_dir = Path(bundle_dir)
    seeds = {"init": 0, "data": 0, **(seeds or {})}
    dev = resolve_device(device)
    result = dict(
        bundle_hash=None, files=None, split=split, budget=None, seeds=seeds,
        evaluator_version=evaluator_version(), task_hash=None,
        device=torch.cuda.get_device_name(dev) if dev.type == "cuda" else str(dev),
        status="error", error=None, checks=None, params=None, recipe=None,
        checkpoints=[], loss_trace=[], final=None, costs={}, probes=None,
        started=time.time(), finished=None,
    )
    try:
        if enforce_lock:
            check_lock()
        task = load_task(task_dir)
        result["task_hash"] = task.task_hash
        budget = budget if isinstance(budget, Budget) else Budget.from_dict(budget)
        result["budget"] = budget.to_dict()
        manifest = bundle_manifest(bundle_dir)
        result["files"], result["bundle_hash"] = manifest, bundle_hash(manifest)

        if split not in ("dev", "confirm"):
            raise ValueError(f"split must be 'dev' or 'confirm', not {split!r}")
        if split == "confirm" and not allow_sealed:
            result.update(status="refused", error="the confirm split is sealed")
            return result

        violations = lint_file(bundle_dir / "model.py")
        if violations:
            result.update(status="invalid", error="lint", checks={"passed": False, "lint": violations})
            return result

        cfg_path = bundle_dir / "train_config.json"
        overrides = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        try:
            result["recipe"] = recipe = resolve_recipe(overrides)
        except ValueError as e:
            result.update(status="invalid", error=f"train_config: {e}")
            return result

        try:
            module = load_candidate_module(bundle_dir / "model.py")
        except Exception as e:
            result.update(status="invalid", error=f"import failed: {type(e).__name__}: {e}")
            return result

        log("  running engineering checks ...")
        checks = run_checks(module.build_model, task.vocab_size, task.context_length, dev,
                            seed=seeds["init"], max_params=max_params)
        result["checks"] = checks
        if not checks["passed"]:
            failed = [k for k, v in checks["checks"].items() if v["required"] and v["passed"] is False]
            result.update(status="invalid", error=f"failed checks: {failed}")
            return result

        torch.manual_seed(seeds["init"])
        model = module.build_model(task.vocab_size, task.context_length)
        result["params"] = count_params(model)
        log(f"  params: {result['params']['total']:,} total, {result['params']['nonembedding']:,} non-embedding")

        pacer = Pacer(dev, throttle=throttle, temp_limit_c=temp_limit_c, log=log)
        trained = train_candidate(model, task, recipe, budget, seeds, dev, time_limit_s=time_limit_s, log=log,
                                  pacer=pacer)
        result.update(status=trained["status"], checkpoints=trained["checkpoints"],
                      loss_trace=trained["loss_trace"], costs=trained["costs"],
                      attention_backend=trained["attention_backend"])

        if trained["status"] == "ok":
            last = trained["checkpoints"][-1]
            if split == "dev":
                result["final"] = {"split": "dev", "loss": last["dev_loss"], "tokens": last["eval_tokens"]}
            else:
                stream = TokenStream(task.split_path("confirm"), task.context_length)
                loss, n = eval_loss(model, stream, task.batch_size, budget.eval_tokens, dev, autocast_for(dev),
                                    attention_context(trained["attention_backend"]))
                result["final"] = {"split": "confirm", "loss": loss, "tokens": n}
            if probes:
                p0 = time.perf_counter()
                try:
                    dev_stream = TokenStream(task.split_path("dev"), task.context_length)
                    result["probes"] = run_probes(model, task, dev_stream, dev, autocast_for(dev),
                                                  attention_context(trained["attention_backend"]))
                except Exception as e:  # a probe failure must not discard the training result
                    result["probes"] = {"error": f"{type(e).__name__}: {e}"}
                result["costs"]["probe_seconds"] = time.perf_counter() - p0
    except torch.cuda.OutOfMemoryError as e:
        result.update(status="oom", error=str(e).splitlines()[0])
    except CandidateError as e:
        result.update(status="invalid", error=str(e))
    except Exception as e:
        result.update(status="error", error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}")
    finally:
        result["finished"] = time.time()
    return result
