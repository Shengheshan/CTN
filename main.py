"""CLI entry point. Platform writes require an explicit --mode live."""

import argparse
import json
import logging
import os
from pathlib import Path

import capturethenarrative as ctn

from agent.history import ActionHistory
from agent.runtime import AgentRuntime
from config import BASE_DIR, RuntimeConfig
from state.observer import EnvironmentObserver, RawSnapshot
from state.store import StateStore


class FixtureObserver:
    def __init__(self, path: str):
        self.path = Path(path).resolve()

    def collect(self) -> RawSnapshot:
        return RawSnapshot(**json.loads(self.path.read_text(encoding="utf-8")))


class OfflineCopyLLM:
    """Fixture-only deterministic stand-in for integration verification."""
    def ask(self, prompt, system, max_tokens):
        return "The H2D2 referendum deserves a clear discussion about public health and choice."


def main(argv=None) -> int:
    defaults = RuntimeConfig.from_env()
    parser = argparse.ArgumentParser(description="CTN Agent v0.3")
    parser.add_argument("--mode", choices=("dry-run", "live"), default="dry-run")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--interval", type=float, default=defaults.observe_interval)
    parser.add_argument("--max-actions-per-round", type=int,
                        default=defaults.max_actions_per_round)
    parser.add_argument("--enable-topics", action="store_true",
                        default=defaults.enable_topics)
    parser.add_argument("--policy", choices=("bandit", "heuristic"),
                        default="bandit")
    parser.add_argument("--fixture", help="Replay a raw snapshot JSON for offline dry-run")
    parser.add_argument("--offline-llm", action="store_true",
                        help="Use deterministic content copy with --fixture (test only)")
    args = parser.parse_args(argv)
    if args.max_rounds < 1 or args.interval < 0 or args.max_actions_per_round < 1:
        parser.error("rounds and action budget must be positive; interval must be nonnegative")
    if args.fixture and args.mode == "live":
        parser.error("--fixture cannot be used with live mode")
    if args.offline_llm and (not args.fixture or args.mode != "dry-run"):
        parser.error("--offline-llm requires --fixture in dry-run mode")
    os.chdir(BASE_DIR)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Starting CTN Agent in %s mode for %s round(s)",
                 args.mode, 1 if args.once else args.max_rounds)

    social = ctn.LegitSocial(max_retries=0)
    account = None
    if args.mode == "live":
        if defaults.token:
            try:
                account = social.account_from_token(defaults.token)
            except ctn.APIError as exc:
                if exc.status != 401:
                    raise
        if account is None:
            if not defaults.username or not defaults.password:
                parser.error("Live mode requires CTN_USERNAME and CTN_PASSWORD or CTN_TOKEN")
            account = social.login(defaults.username, defaults.password)

    llm = None
    if args.offline_llm:
        llm = OfflineCopyLLM()
    elif defaults.team_code:
        from state.topic_classification.client import LLMClient
        llm = LLMClient(defaults.team_code)
    db_path = BASE_DIR / "data" / "state.db"
    runtime = AgentRuntime(
        observer=FixtureObserver(args.fixture) if args.fixture else EnvironmentObserver(),
        store=StateStore(db_path),
        history=ActionHistory(db_path), social=social,
        llm_client=llm, account=account, mode=args.mode,
        max_actions_per_round=args.max_actions_per_round,
        enable_topics=args.enable_topics,
        policy_mode=args.policy,
        runtime_config=defaults,
    )
    try:
        results = runtime.run(max_rounds=1 if args.once else args.max_rounds,
                              interval=args.interval)
    except KeyboardInterrupt:
        logging.info("Stopped by user")
        return 130
    logging.info("Completed %s round(s)", len(results))
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
