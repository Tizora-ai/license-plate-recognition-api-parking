"""
CLI for local predict and latency benchmarking.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# Allow running as: python python/cli.py ...
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.alpr_engine import engine_info, predict_image, warmup  # noqa: E402


def cmd_predict(args: argparse.Namespace) -> int:
    path = Path(args.image)
    if not path.is_file():
        print(f"Image not found: {path}", file=sys.stderr)
        return 1
    warmup(runs=1)
    plates, latency_ms = predict_image(str(path))
    print(json.dumps({"plates": plates, "latency_ms": round(latency_ms, 2)}, indent=2))
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    path = Path(args.image)
    if not path.is_file():
        print(f"Image not found: {path}", file=sys.stderr)
        return 1
    warmup(runs=args.warmup)
    samples: list[float] = []
    last_plates: list = []
    for _ in range(args.runs):
        plates, latency_ms = predict_image(str(path))
        samples.append(latency_ms)
        last_plates = plates
    samples_sorted = sorted(samples)
    p50 = statistics.median(samples_sorted)
    p95_idx = min(len(samples_sorted) - 1, int(round(0.95 * (len(samples_sorted) - 1))))
    p95 = samples_sorted[p95_idx]
    mean_ms = statistics.fmean(samples_sorted)
    print(
        json.dumps(
            {
                "info": engine_info(),
                "runs": args.runs,
                "warmup": args.warmup,
                "mean_ms": round(mean_ms, 2),
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "min_ms": round(min(samples_sorted), 2),
                "max_ms": round(max(samples_sorted), 2),
                "last_plates": last_plates,
            },
            indent=2,
        )
    )
    return 0


def cmd_info(_: argparse.Namespace) -> int:
    print(json.dumps(engine_info(), indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="alpr CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_predict = sub.add_parser("predict", help="Recognize plates in an image")
    p_predict.add_argument("image", help="Path to image")
    p_predict.set_defaults(func=cmd_predict)

    p_bench = sub.add_parser("bench", help="Latency benchmark")
    p_bench.add_argument("image", help="Path to image")
    p_bench.add_argument("--runs", type=int, default=20)
    p_bench.add_argument("--warmup", type=int, default=3)
    p_bench.set_defaults(func=cmd_bench)

    p_info = sub.add_parser("info", help="Show engine / provider info")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
