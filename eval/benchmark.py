"""Backward-compatible entry point: ``python -m eval.benchmark`` → :mod:`argmine.evaluate`."""
from argmine.evaluate import align, evaluate, load_jsonl, main, prf

__all__ = ["align", "evaluate", "load_jsonl", "main", "prf"]

if __name__ == "__main__":
    main()
