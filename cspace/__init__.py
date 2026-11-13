"""cspace: a research harness for transformational-creativity experiments.

Gate 1 components:
  cspace.data       corpus preparation and deterministic token streams
  cspace.evaluator  locked evaluator: lint, engineering checks, compute-capped training
  cspace.archive    content-addressed artifact store, SQLite index, cost ledgers
  cspace.box        K/C dependency-graph records (validation only, for Gate 2)
"""

__version__ = "0.1.0"
