# CTN Agent v0.2 technical design and verification

## Scope and data flow

The runtime runs a bounded loop:

`observe → normalize → save snapshot → build WorldState → encode → load AgentState → generate candidates → score → select ActionPlan → generate and validate content → execute → record result and feedback`.

The existing `state/` pipeline is reused. The `policy/` package now returns
outward actions instead of inspection tasks. `agent/` owns persistence,
execution, feedback and orchestration. `main.py` is the CLI entry point.

## Contracts

CandidateAction includes action type, target ID, source event ID, context,
rationale, estimated cost and required parameters. ActionPlan adds a stable
action ID, snapshot ID, score and score components, reason, parameters,
generated content, status and creation time. The engine never calls the SDK.

Action types: CREATE_POST, REPLY, LIKE, REPOST, FOLLOW, WAIT. Observation is a
runtime step. The installed SDK maps these to Account.post, reply, like,
repost and follow. The SDK has no separate comment method; comments beneath
posts use reply.

Candidate filters require observed text for replies and attributed article
titles for posts. Endorsement and relationship actions require an explicit
positive alignment signal. The current runtime supplies no such signal and
does not infer agreement from popularity. This limits normal automatic
selection to sourced posts, neutral replies and WAIT. Action eligibility also
uses persisted history, account likes/reposts when available, feed lock and
per-round budget.

The heuristic considers relevance, freshness, heat, trend, news evidence,
data completeness, quota, repetition and cost. All weights are in
`policy/settings.py`. Missing task relevance defaults to a neutral value;
scores are not impact probabilities or learned rewards. A separate WAIT
baseline permits no action.

## Content and execution

Only CREATE_POST and REPLY call the competition LLM. The prompt asks the model
to copy one exact source title or a bounded excerpt of the target post. Output
must match the supplied text exactly; the final message is an attributed quote
or a neutral request for context. Empty, invented or oversized output stops
the action. This narrow method reduces unsupported claims but does not prove
the underlying source is true. Topic classification is never a prerequisite.

Dry-run constructs and validates content, records DRY_RUN and performs no
platform write. Live mode requires an explicit `--mode live`, an existing
account token or username/password, and a feed-lock check before each write.
The SDK client is initialized with `max_retries=0`; the executor handles a 429
with at most one bounded Retry-After wait and retry. Lost responses to writes
are marked UNKNOWN and block duplicate action keys until resolved. The SQLite
journal reserves actions before writes and records remote IDs and feedback.

## Commands

From `D:\CTN-bot`:

```powershell
.\.venv\Scripts\python.exe -B -m unittest -v test_policy test_action_policy test_v02
.\.venv\Scripts\python.exe -B main.py --mode dry-run --once
.\.venv\Scripts\python.exe -B main.py --mode dry-run --once --fixture data\raw\snapshot_20261031_162610_663328.json --offline-llm
.\.venv\Scripts\python.exe -B main.py --mode live --once
.\.venv\Scripts\python.exe -B main.py --mode live --max-rounds 10 --interval 60 --max-actions-per-round 1
```

`--fixture --offline-llm` is an offline integration check; it cannot be used
with live mode. `--enable-topics` opts into the existing classifier. Default
mode is dry-run, and max rounds defaults to one. Live mode needs CTN_USERNAME
and CTN_PASSWORD or CTN_TOKEN in `.env`; CTN_TEAM_CODE is needed for real LLM
content generation. `.gitignore` excludes `.env`, tokens, databases and raw
snapshots.

## Verification and remaining limits

On 2026-11-01, 12 focused tests passed in the project environment, and the
existing encoder test passed. Mock SDK tests cover every action mapping,
dry-run write exclusion, restart deduplication, duplicate content, feed lock,
401/403/409, bounded 429 retry and unknown timeout. Fake observer and LLM
tests cover two complete rounds, snapshot storage, encoding, decision, content
validation and action journaling. The fixture dry-run exercised the CLI with
45 posts, 20 articles and 13 events and produced CREATE_POST / DRY_RUN.

A real `--mode dry-run --once` also completed: platform observation saved
Snapshot 3 with 45 posts, 20 articles and 14 events; the competition LLM
responded successfully; the journal recorded CREATE_POST / DRY_RUN and one
feedback row. No platform write was sent. Live write behavior is verified only
with Mock SDK because the current `.env` contains a team code but no account
username, password or token. The real live mode has not been run.

Remaining limits: current topic classification is semantically unreliable;
automatic alignment for LIKE/REPOST/FOLLOW is unavailable; content generation
is deliberately narrow and can skip an action if the model changes source
text; no causal reward or Contextual UCB is implemented; uncertain writes
require manual resolution. Per-route API quota information is not exposed as a
complete snapshot by this SDK, so runtime relies on conservative intervals,
the SDK rate limit response and bounded 429 handling.
