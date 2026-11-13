# Running evaluations on cloud GPUs or a cluster

Everything that trains a model goes through `cspace.executor`. To use remote GPUs,
write one `Executor` subclass that moves a packed request to the remote machine and
brings the result back. The rest is already in place:

- `parallel_jobs` runs a wave's jobs at the same time, and `executor.slots` sets how
  many evaluations run at once.
- The calibration cohort and confirmations run their members concurrently.
- Each result records the executor, a hashed hostname, the torch and CUDA versions, and
  the GPU name.
- `LoopbackExecutor` already runs the whole remote path locally (pack, unpack, find the
  task by hash, isolated run), and a test checks that its results are bitwise identical
  to local runs.

The agent loop, the archive and `OPENAI_API_KEY` stay on your machine. The remote side
receives only model bundles (a few KB each) and needs no keys.

## What the remote machine needs

1. This repository, at the same commit. Results carry the evaluator version, and
   `freeze-evaluator` locks it.
2. The same Python dependencies (`pip install -e .`).
3. The prepared task. Run `python -m cspace prepare-data --corpus tinystories` there.
   The task is found by content hash, so it must match the local one. `find_task`
   refuses a mismatch.

The remote entry point is either of:

```
python -m cspace.executor run-packed packed.json result.json --task-root data/
```

```python
from cspace.executor import execute_packed
result = execute_packed(packed, task_root="data/")
```

## Writing the backend (the one piece left)

```python
# cspace/executors_cloud.py  (sketch, not yet written or tested)
from cspace.executor import Executor, pack_request, register_executor

@register_executor
class ModalExecutor(Executor):
    kind = "modal"

    def _execute(self, request, timeout_s):
        packed = pack_request(request)
        # e.g. a Modal function with gpu="L4" whose image has the repo and data/ baked in:
        #     return remote_fn.remote(packed)   # remote_fn calls execute_packed(packed, "/root/data")
        # or for SLURM: write packed.json over ssh/scp, sbatch a script that runs
        # `python -m cspace.executor run-packed ...`, poll squeue, fetch result.json.
        raise NotImplementedError
```

Import the module before creating campaigns (for example from `cspace/cli.py`). Then:

```
python -m cspace campaign init pilot-dag --executor modal --slots 4 --parallel-jobs 4
python -m cspace cohort init calib1 --source pilot-dag --executor modal --slots 8
python -m cspace evaluate --candidate <id> --executor modal ...
```

`run()` must return the evaluator's result dict. On a transport failure, return
`{"status": "error", "error": "...", "started": t, "finished": t}`; don't raise. The job
then reports an infrastructure error instead of crashing the wave.

## Before a study on remote hardware

1. Use the loopback first: `python -m cspace evaluate --candidate <seed> --executor loopback`.
2. Run the baseline twice on the remote GPU type, then `replay --tolerance 0`. Remote
   runs must be bitwise reproducible, as local runs are.
3. Use one GPU type for every arm and pair of a study. FLOP budgets don't depend on
   the device, but losses can differ slightly between GPU models. The `device` field
   of each result records which GPU ran it.
4. Freeze the evaluator (`python -m cspace freeze-evaluator`) and commit before
   launching campaigns.
5. Set the cost caps. The ledgers record training device-seconds next to FLOPs.
