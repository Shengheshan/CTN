"""Collect read-only post-counter samples at ages relative to publication."""

import argparse
import json
import time

import capturethenarrative as ctn

from agent.observation_monitor import ObservationMonitor
from config import BASE_DIR


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect CTN interaction counters at fixed post-age windows")
    parser.add_argument("--watch", action="store_true",
                        help="Keep checking due windows until Ctrl+C")
    parser.add_argument("--interval", type=float, default=60.0,
                        help="Seconds between due-window checks")
    parser.add_argument("--limit", type=int, default=500,
                        help="Maximum recent successful posts/replies per check")
    args = parser.parse_args(argv)
    if args.interval < 10 or args.limit < 1:
        parser.error("interval must be at least 10 seconds and limit positive")

    monitor = ObservationMonitor(BASE_DIR / "data" / "state.db",
                                 ctn.LegitSocial(max_retries=0))
    try:
        while True:
            samples = monitor.observe_due_windows(
                limit=args.limit, poll_interval_seconds=args.interval)
            for sample in samples:
                print(json.dumps({"kind": "feedback_window", **sample},
                                 ensure_ascii=False), flush=True)
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
