"""Hardware-protective pacing for training runs.

- throttle in [0, 1): after each step, sleep long enough that the device is busy
  only (1 - throttle) of the wall-clock time. 0.5 halves sustained load.
- temp_limit_c: every `check_every` steps, read the GPU temperature (nvidia-smi);
  at or above the limit, pause until it drops 5 C below the limit. A run that stays
  hot longer than max_pause_s stops with status "overheat".

Pacing changes wall-clock time only; the computation, and therefore every
measurement, is identical with or without it.
"""

import subprocess
import time

import torch


def gpu_temperature(index: int = 0):
    """Current GPU temperature in C, or None if it cannot be read."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits",
                              "-i", str(index)], capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


class Overheat(RuntimeError):
    pass


class Pacer:
    def __init__(self, device, throttle=0.0, temp_limit_c=None, check_every=50, max_pause_s=1800.0, log=print):
        if not 0.0 <= float(throttle) < 1.0:
            raise ValueError("throttle must be in [0, 1)")
        self.device = torch.device(device)
        self.throttle = float(throttle)
        self.temp_limit = temp_limit_c if self.device.type == "cuda" else None
        self.check_every = check_every
        self.max_pause_s = max_pause_s
        self.log = log
        self.paused_seconds = 0.0
        self.temp_pauses = 0
        self.max_temp_seen = None
        self._t = None

    @property
    def active(self):
        return self.throttle > 0 or self.temp_limit is not None

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def step_start(self):
        if self.throttle > 0:
            self._sync()
            self._t = time.perf_counter()

    def step_end(self, step: int):
        if self.throttle > 0 and self._t is not None:
            self._sync()
            busy = time.perf_counter() - self._t
            self._sleep(busy * self.throttle / (1.0 - self.throttle))
        if self.temp_limit is not None and step % self.check_every == 0:
            self._check_temperature()

    def _sleep(self, seconds):
        if seconds > 0:
            time.sleep(seconds)
            self.paused_seconds += seconds

    def _check_temperature(self):
        temp = gpu_temperature()
        if temp is None:
            return
        self.max_temp_seen = temp if self.max_temp_seen is None else max(self.max_temp_seen, temp)
        if temp < self.temp_limit:
            return
        self.temp_pauses += 1
        self.log(f"  GPU at {temp} C (limit {self.temp_limit} C): pausing to cool")
        start = time.perf_counter()
        while temp is not None and temp > self.temp_limit - 5:
            if time.perf_counter() - start > self.max_pause_s:
                raise Overheat(f"GPU stayed at or above {self.temp_limit - 5} C for {self.max_pause_s:.0f}s")
            self._sleep(5.0)
            temp = gpu_temperature()

    def report(self):
        return {"throttle": self.throttle, "temp_limit_c": self.temp_limit, "paused_seconds": self.paused_seconds,
                "temp_pauses": self.temp_pauses, "max_temp_seen_c": self.max_temp_seen}
