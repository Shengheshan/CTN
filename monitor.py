"""Observe action history and platform counters; never publish or train."""

import argparse
import json
import time

import capturethenarrative as ctn

from agent.observation_monitor import ObservationMonitor
from config import BASE_DIR


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read-only CTN action monitor")
    parser.add_argument("--watch", action="store_true", help="Keep polling until Ctrl+C")
    parser.add_argument("--interval", type=float, default=60.0,
                        help="Seconds between samples in watch mode")
    parser.add_argument("--limit", type=int, default=20,
                        help="Maximum recent actions/posts to inspect per sample")
    args = parser.parse_args(argv)
    if args.interval < 10 or args.limit < 1:
        parser.error("interval must be at least 10 seconds and limit positive")
    monitor = ObservationMonitor(BASE_DIR / "data" / "state.db",
                                 ctn.LegitSocial(max_retries=0))
    try:
        while True:
            for action in monitor.recent_actions(args.limit):
                print(json.dumps({"kind": "action", **action}, ensure_ascii=False), flush=True)
            for observation in monitor.observe_once(args.limit):
                print(json.dumps({"kind": "counter_observation", **observation},
                                 ensure_ascii=False), flush=True)
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
