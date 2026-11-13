"""Run an evaluation in a separate process with a hard timeout.

The child process gets a scrubbed environment (no API keys or tokens), so code
written by agents cannot read credentials, and a crash or hang in candidate code
cannot take down the orchestrator.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from ..paths import ROOT

_SECRET_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def scrubbed_env():
    env = {k: v for k, v in os.environ.items() if not any(m in k.upper() for m in _SECRET_MARKERS)}
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_isolated(request: dict, timeout_s: float, stream_output=True) -> dict:
    request = {k: (str(v) if isinstance(v, Path) else v) for k, v in request.items()}
    with tempfile.TemporaryDirectory(prefix="cspace_eval_") as tmp:
        req_path, res_path = Path(tmp) / "request.json", Path(tmp) / "result.json"
        req_path.write_text(json.dumps(request), encoding="utf-8")
        started = time.time()
        cmd = [sys.executable, "-m", "cspace.evaluator.worker", str(req_path), str(res_path)]
        proc = subprocess.Popen(cmd, cwd=tmp, env=scrubbed_env(), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        lines = []

        def pump():
            for line in proc.stdout:
                lines.append(line)
                if stream_output:
                    print(line, end="", flush=True)

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            reader.join(timeout=5)
            return {"status": "timeout", "error": f"hard timeout after {timeout_s:.0f}s",
                    "started": started, "finished": time.time(), "log_tail": "".join(lines[-40:])}
        reader.join(timeout=5)
        if res_path.is_file():
            result = json.loads(res_path.read_text(encoding="utf-8"))
        else:
            result = {"status": "error", "error": f"worker exited with code {proc.returncode} without a result",
                      "started": started, "finished": time.time()}
        result["log_tail"] = "".join(lines[-40:])
        return result
