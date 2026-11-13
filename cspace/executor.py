"""Where evaluations run.

Everything that trains a candidate (campaigns, the calibration cohort, replication
and confirmation, the `evaluate` command) goes through an Executor. An evaluation
request is a small JSON-able dict (bundle directory, task directory, split, budget,
seeds, device, limits); the result is the evaluator's JSON result.

    executor = get_executor({"kind": "local", "slots": 1})
    result = executor.run(request)                          # thread-safe, blocks
    executor.run_all([(key, request), ...], on_result)      # concurrent, callback on caller's thread

`slots` bounds how many evaluations run at once through one executor; callers may
submit from many threads (parallel wave jobs, cohort members) and the executor
queues them. Executors are shared per spec within a process, so the bound holds
across campaign threads.

Adding a remote backend (cloud GPUs, a SLURM cluster) means writing one subclass:

    class MyCloudExecutor(Executor):
        kind = "mycloud"
        def _execute(self, request, timeout_s):
            packed = pack_request(request)          # self-contained: bundle files inlined, task by hash
            ...send `packed` (JSON) to the remote machine, which runs
               `python -m cspace.executor run-packed packed.json result.json --task-root <data dir>`
               (or calls execute_packed() directly), and fetch result.json...
            return result
    register_executor(MyCloudExecutor)

The remote machine needs this repository (same commit: the evaluator version is
checked against the result) and the prepared task data. `LoopbackExecutor` runs
exactly that remote path on this machine, so the packing and unpacking are tested
without any cloud access. See CLOUD.md.
"""

import argparse
import base64
import json
import platform
import socket
import tempfile
import threading
import time
from pathlib import Path

from .hashing import bundle_hash, bundle_manifest

_REGISTRY = {}
_INSTANCES = {}
_LOCK = threading.Lock()
TIMEOUT_GRACE_S = 600  # hard timeout = the evaluator's own time limit + this


def build_request(bundle_dir, task_dir, settings: dict, budget: dict, seeds: dict, split="dev",
                  allow_sealed=False) -> dict:
    """Evaluation request from a campaign/cohort/confirmation settings dict (device, throttle,
    temp_limit_c, eval_time_limit_s, max_params)."""
    return dict(bundle_dir=str(bundle_dir), task_dir=str(task_dir), split=split, allow_sealed=allow_sealed,
                budget=budget, seeds=seeds, device=settings.get("device", "auto"),
                time_limit_s=settings["eval_time_limit_s"], max_params=int(settings["max_params"]),
                throttle=settings.get("throttle", 0.0), temp_limit_c=settings.get("temp_limit_c"))


def register_executor(cls):
    _REGISTRY[cls.kind] = cls
    return cls


def normalize_spec(spec) -> dict:
    if spec is None:
        spec = "local"
    if isinstance(spec, str):
        spec = {"kind": spec}
    spec = dict(spec)
    spec.setdefault("kind", "local")
    spec.setdefault("slots", 1)
    if spec["kind"] not in _REGISTRY:
        raise ValueError(f"unknown executor {spec['kind']!r}; registered: {sorted(_REGISTRY)}")
    if int(spec["slots"]) < 1:
        raise ValueError("executor slots must be >= 1")
    return spec


def get_executor(spec=None) -> "Executor":
    """Shared executor instance for this spec (one slot pool per spec per process)."""
    spec = normalize_spec(spec)
    key = json.dumps(spec, sort_keys=True)
    with _LOCK:
        if key not in _INSTANCES:
            options = {k: v for k, v in spec.items() if k not in ("kind", "slots")}
            _INSTANCES[key] = _REGISTRY[spec["kind"]](slots=int(spec["slots"]), **options)
        return _INSTANCES[key]


class Executor:
    kind = "base"

    def __init__(self, slots=1, **options):
        self.slots = int(slots)
        self.options = options
        self._sem = threading.BoundedSemaphore(self.slots)

    def describe(self) -> dict:
        return {"kind": self.kind, "slots": self.slots, **self.options}

    def run(self, request: dict, timeout_s: float = None) -> dict:
        request = {k: (str(v) if isinstance(v, Path) else v) for k, v in request.items()}
        timeout_s = timeout_s or float(request.get("time_limit_s") or 3600) + TIMEOUT_GRACE_S
        t0 = time.time()
        with self._sem:
            queued = time.time() - t0
            result = self._execute(request, timeout_s)
        result["executor"] = {**self.describe(), "queue_seconds": round(queued, 3)}
        return result

    def run_all(self, jobs, on_result):
        """Run [(key, request), ...] concurrently (bounded by slots). `on_result(key, result)` is
        called on the calling thread as each finishes, so callers can record results with
        thread-bound resources (SQLite) and keep finished work if interrupted."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not jobs:
            return
        with ThreadPoolExecutor(max_workers=min(self.slots, len(jobs))) as pool:
            futures = {pool.submit(self.run, request): key for key, request in jobs}
            for f in as_completed(futures):
                on_result(futures[f], f.result())

    def _execute(self, request: dict, timeout_s: float) -> dict:
        raise NotImplementedError


@register_executor
class LocalExecutor(Executor):
    """Isolated subprocess on this machine (scrubbed environment, hard timeout)."""
    kind = "local"

    def _execute(self, request, timeout_s):
        from .evaluator.runner import run_isolated

        result = run_isolated(request, timeout_s=timeout_s, stream_output=bool(self.options.get("stream")))
        result["host"] = host_info()
        return result


@register_executor
class LoopbackExecutor(Executor):
    """The remote code path, run locally: pack -> JSON -> unpack in a fresh directory
    with the task located by hash under `task_root` -> isolated run. Use it to test a
    remote setup end to end; a real remote executor only replaces the transport."""
    kind = "loopback"

    def _execute(self, request, timeout_s):
        packed = json.loads(json.dumps(pack_request(request)))
        task_root = self.options.get("task_root") or str(Path(request["task_dir"]).parent)
        return execute_packed(packed, task_root, timeout_s=timeout_s)


# ---------------------------------------------------------------------------
# self-contained requests for remote execution
PACK_VERSION = 1


def pack_request(request: dict) -> dict:
    """Inline the bundle and replace the task path by its hash, so the request can be
    shipped to a machine that has the repository and the task data but not this archive."""
    from .task import load_task

    bundle_dir = Path(request["bundle_dir"])
    manifest = bundle_manifest(bundle_dir)
    files = {name: base64.b64encode((bundle_dir / name).read_bytes()).decode("ascii") for name in manifest}
    task = load_task(request["task_dir"])
    body = {k: v for k, v in request.items() if k not in ("bundle_dir", "task_dir")}
    return {"pack_version": PACK_VERSION, "bundle_files": files, "bundle_hash": bundle_hash(manifest),
            "task_hash": task.task_hash, "task_corpus": task.corpus, "request": body}


def find_task(task_root, task_hash: str) -> Path:
    from .task import load_task

    root = Path(task_root)
    candidates = [root] if (root / "task.json").is_file() else sorted(p.parent for p in root.glob("*/task.json"))
    for d in candidates:
        if load_task(d).task_hash == task_hash:
            return d
    raise FileNotFoundError(f"no task with hash {task_hash[:12]} under {root}; run prepare-data there")


def unpack_request(packed: dict, task_root, workdir) -> dict:
    if packed.get("pack_version") != PACK_VERSION:
        raise ValueError(f"unsupported pack version {packed.get('pack_version')}")
    bundle = Path(workdir) / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    for name, data in packed["bundle_files"].items():
        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(f"unsafe bundle file name {name!r}")
        (bundle / name).write_bytes(base64.b64decode(data))
    if bundle_hash(bundle_manifest(bundle)) != packed["bundle_hash"]:
        raise ValueError("bundle hash mismatch after unpacking")
    return {**packed["request"], "bundle_dir": str(bundle), "task_dir": str(find_task(task_root, packed["task_hash"]))}


def execute_packed(packed: dict, task_root, timeout_s: float = None) -> dict:
    """Remote-side entry point: run one packed request in an isolated subprocess."""
    from .evaluator.runner import run_isolated

    timeout_s = timeout_s or float(packed["request"].get("time_limit_s") or 7200) + TIMEOUT_GRACE_S
    with tempfile.TemporaryDirectory(prefix="cspace_remote_") as tmp:
        request = unpack_request(packed, task_root, tmp)
        result = run_isolated(request, timeout_s=timeout_s, stream_output=False)
    result["host"] = host_info()
    return result


def host_info() -> dict:
    info = {"hostname_hash": _short_hash(socket.gethostname()), "platform": platform.platform(),
            "python": platform.python_version()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
    except Exception:
        pass
    return info


def _short_hash(text):
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()[:8]


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m cspace.executor", description="remote-side evaluation entry point")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run-packed", help="run a packed request and write the result JSON")
    r.add_argument("packed")
    r.add_argument("result")
    r.add_argument("--task-root", required=True, help="directory containing prepared task(s)")
    r.add_argument("--timeout", type=float)
    args = p.parse_args(argv)
    packed = json.loads(Path(args.packed).read_text(encoding="utf-8"))
    result = execute_packed(packed, args.task_root, timeout_s=args.timeout)
    Path(args.result).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
