# CTN Agent v0.3 — Policy Evaluation & Feedback

## Scope and safety properties

v0.3 adds decision/candidate audit rows, deterministic content assessments, scheduled feedback snapshots, versioned action-quality/owned-feedback rewards, opt-in contextual LinUCB learning, and offline reports. It does not use vote or poll changes as reward. Engagement counters can contribute only when they belong to the action's own post or reply; parent-post changes are analysis-only.

Existing v0.2 tables and records are retained. New SQLite tables are created with `CREATE TABLE IF NOT EXISTS`; no historical action rows are deleted or rewritten. The old engagement-proxy tables remain for audit but are no longer loaded by `AgentRuntime` as a v0.3 model.

## Decision and candidate log

`policy_decisions` stores one row for the decision trace: snapshot, policy/encoder versions, selected action, selection method, candidate count, and complete trace JSON. `candidate_scores` stores one row per evaluated candidate, including base score, configured Action Prior, cost/risk penalties, exploration bonus, final selection score, selected flag, policy features, opportunity evidence, and score components. Default Action Priors are neutral `1.0`, preserving the prior stable scoring behavior. Override them with `CTN_ACTION_PRIORS` as a JSON map, for example `{"CREATE_POST":1.0,"REPLY":1.0,"LIKE":1.0}`. Runtime never automatically changes configured priors unless learning is enabled.

Decision Score (pre-action estimate), Exploration Bonus, and Observed Reward are separate fields. Exploration Bonus is retained in the candidate trace and never added to an action reward.

## Content assessments

`policy.novelty` compares a normalized exact hash and word 3-gram Jaccard similarity against the most recent successful authored actions. `novelty_score = 1 - max_similarity`; short content and missing comparable history yield `UNKNOWN`/NULL, not a high score. The assessment records algorithm version and comparison-window size. The deterministic lexical heuristic may miss paraphrases and is not a content-quality judgment.

`policy.source_validator` matches draft URLs only against supplied, persisted article records. It removes the appended attribution from lexical comparison, reports unmatched sources and numeric claims, and marks a numeric-context mismatch `CONTRADICTED`. A match or lexical overlap is never claimed to verify truth; `fact_check_status` remains `UNKNOWN` unless the conservative numeric conflict check fires. `source_alignment_score` is capped at 0.5 and is only a lexical source-alignment signal.

Assessments are stored in `content_assessments` and embedded in action parameters for traceability.

## Feedback schedule and missingness

`action_feedback_snapshots` schedules `CTN_FEEDBACK_WINDOWS` seconds after the action's actual saved execution time. Default windows are 300, 1800, 7200, and 21600 seconds (5 min, 30 min, 2 h, 6 h). `(action_id,window_name)` is unique. Runtime checks due rows once per round and resumes pending rows after restart.

For CREATE_POST and REPLY, it reads counters on the action's own remote object. It does not attribute parent-post counters to a reply. LIKE, REPOST, FOLLOW, and MENTION retain execution status; when the platform has no action-specific counter endpoint, their feedback window is `UNAVAILABLE`, not inferred from the target's popularity. WAIT records a `next_observation` row and reason. Missing metric columns remain NULL. Deltas are only calculated when both readings exist; interval duration is stored. HTTP 429 responses are retried at most three times using the server delay (bounded to 5 minutes); 404 becomes UNAVAILABLE and other errors become FAILED.

Feedback collection exceptions are caught in Runtime and do not stop the observation loop. It uses read-only counter calls; no additional platform writes are made.

## Reward and learning

`reward-v1-quality` initially uses the configured weights (defaults E=.30, F=.35, N=.20, P=.15):

- E: action execution success.
- F: lexical source alignment, only when a news source applies; it is not a full fact check.
- N: assessed novelty, only for text actions with comparable history.
- P: immediate platform acceptance; this does not prove the object was never later removed.

Non-applicable/unknown components are omitted and remaining weights renormalized; calculation JSON records applicable, used, and missing dimensions. Reward begins PENDING, becomes PROVISIONAL while scheduled windows remain, and becomes FINAL after all windows resolve. Failed/blocked actions are scored from known execution status; UNKNOWN execution is INVALID. Different reward versions and recalculations are append-only and retain their full calculation JSON. Own-object counters can contribute to v2 observed feedback; parent-post counters do not contribute to REPLY reward.

`CTN_ENABLE_LEARNING` defaults to false in code. When enabled, final rewards for CREATE_POST and REPLY train the versioned 32-feature LinUCB from the selected candidate vector already stored in `candidate_scores`; the model is persisted in `policy_states` and restored at restart. Bandit mode switches from the heuristic/opportunity Softmax prior to LinUCB + Softmax after five eligible updates. During cold start, the first five final text rewards also adjust bounded action priors; the prior update step defaults to 0.01, baseline decay to 0.02, and prior deviation is capped at 0.20. LIKE, REPOST, FOLLOW, MENTION and WAIT do not train because this API provides no action-specific outcome for them. Old finalized rewards recorded as `learning_disabled_fixed_baseline` can be applied once after learning is enabled; their original audit rows remain intact. `--policy heuristic` continues to use the Softmax heuristic and does not use LinUCB scores.

## Configuration

- `CTN_ENABLE_FEEDBACK=true`
- `CTN_FEEDBACK_WINDOWS=300,1800,7200,21600`
- `CTN_ENABLE_POLICY_ANALYTICS=true`
- `CTN_ENABLE_LEARNING=false` (code default; this local `.env` opts in)
- `CTN_TEXT_ACTIONS_PER_ENGAGEMENT=3`
- `CTN_REWARD_VERSION=reward-v2-action-feedback`
- `CTN_REWARD_WEIGHTS={}` (JSON overrides, all four component names required if any are supplied)
- `CTN_ACTION_PRIORS={}` (JSON, values 0–2; default 1 per action)
- `CTN_NOVELTY_HISTORY_SIZE=100`
- `CTN_NOVELTY_DUPLICATE_THRESHOLD=0.90`

Secrets and account credentials remain in `.env`; reporting never prints them.

## Reports and offline comparison

Run `python -B -m analytics.policy_report --db data/state.db`. It writes CSV tables, JSON summary, PNG dashboard, and a notes file under `data/analysis/`. Empty series are marked “Data insufficient”; no mock observations are inserted. The report includes reward, success rate, action mix, feedback-window counters/missingness, novelty, source-validation status, score/reward comparison, and parameter updates. It reports date range and sample counts. Correlation is descriptive and is not presented as causal impact.

For replaying alternate priors against the same recorded candidate sets, pass a JSON file such as:

```json
{"baseline":{"CREATE_POST":1.0,"REPLY":1.0},"variant":{"CREATE_POST":0.8,"REPLY":1.2}}
```

The resulting `prior_comparison.csv` only shows how the ranking changes on recorded candidates. It has no counterfactual outcomes and cannot demonstrate that a prior improves results.

## Operational notes and limits

The policy evaluation runs from Runtime at the start of each round; no separate feedback-monitor process is required. Do not run the legacy `feedback_monitor.py` alongside v0.3 Runtime unless you deliberately want its separate diagnostic samples and extra read calls. New database tables appear when the updated Runtime opens the DB. Existing feedback collected before v0.3 is not backfilled into scheduled windows because its timing does not match the configured measurement windows.

## Incremental policy/feedback update

`CTN_MINIMUM_POST_COVERAGE` (default `0.55`) sets the minimum share of
successful `CREATE_POST` actions among successful `CREATE_POST` and `REPLY`
actions in the last `CTN_POST_COVERAGE_WINDOW` (default `20`). With no prior
successful text actions, the first selection is unconstrained. When the target
is below threshold and a post candidate is eligible, the scheduler selects
among eligible original-post candidates. `CTN_CREATE_POST_PRIORITY` and
`CTN_REPLY_PRIORITY` (defaults `0.35` and `-0.12`) are score terms only; these
priority terms do not enter reward calculation. WAIT and all existing
candidate/execution gates remain active. A handled news event becomes eligible
again after `CTN_POST_EVENT_COOLDOWN_HOURS` (default `2`), using the snapshot ID
in the idempotency key. Exact duplicate content checks remain in ActionHistory.

Decision traces now retain `content_features`, `target_features`,
`timing_features`, and `environment_features`. Replies capture a parent-post
counter baseline immediately before execution where the API permits. Each
feedback window records counters for both the reply object and its parent post;
the reply's own metrics and the parent's metrics/deltas are separate. Missing
fields remain NULL. Parent-post deltas are never added to reply reward.

The default reward version is now `reward-v2-action-feedback`; v1 rows remain
unchanged. v2 keeps execution, source alignment, and compliance components,
adds reply-context relevance, and can include a bounded feedback component
computed from the action's own object counters. Novelty remains recorded for
analysis/exploration and is excluded from observed reward. The reply feedback
component uses reply-object metrics only.
Decision score, exploration bonus, and observed reward remain separate.
Analytic grouping by reply context/time is descriptive and does not imply
causation. `reply_context_feedback.csv` exports those measurements with an
explicit non-causal attribution label.

Public counters do not identify NPCs, unique people, policy persuasion, or later moderation/removal. No causal claim can be made from these data. News verification is conservative and local; no LLM claim-level fact checker is enabled in this implementation.
