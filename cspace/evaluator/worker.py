"""Subprocess entry point: python -m cspace.evaluator.worker request.json result.json"""

import json
import sys


def main():
    request_path, result_path = sys.argv[1], sys.argv[2]
    from .determinism import configure

    configure(deterministic=True)  # before CUDA initializes
    from .evaluate import evaluate_bundle

    with open(request_path, encoding="utf-8") as f:
        request = json.load(f)
    result = evaluate_bundle(**request, log=lambda msg: print(msg, flush=True))
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
