# CTN Agent policy

`CandidateGenerator` proposes CREATE_POST, REPLY, LIKE, REPOST, FOLLOW, MENTION and WAIT.
`PolicyScorer` evaluates action, target and context with the centralized
`PolicySettings` weights. `DecisionEngine` chooses one ActionPlan.
It never calls the SDK. `agent.executor` owns platform writes.

CREATE_POST and REPLY require a source relevant to the H2D2 referendum.
LIKE, REPOST, FOLLOW and MENTION require an explicit positive alignment signal;
heat alone is insufficient. MENTION uses a relevant published post from this
agent and is limited to one mention per own post and recipient. Topic
classification remains optional. This lexical gate is a conservative action
eligibility check, not a general topic classifier.

The heuristic is an engineering baseline, not a measured impact or reward.
WAIT has its own score and wins ties. Action limits and feed lock can force WAIT.
The competition LLM writes CREATE_POST and REPLY text from a structured brief
containing tone, purpose, target context and timely news sources. Formatting,
length and unsupported-number checks run before execution.
Each round reads the platform's popular tags. Tag-feed observation still reads
two selected feeds to avoid extra API calls. CREATE_POST instead samples up to
three current relevant tags from a wider pool, weighting popularity and reducing
repetition from the last three successful own posts. Context-specific tags
require matching news text; opposing, unrelated, and absent tags are excluded.
The generation budget may remove tags when a source URL is long. Final tags are
saved with the action plan.
The runtime also folds the account's Home feed, recent tag feeds, and unresolved
notification posts into the durable snapshot. A candidate from a notification
gets a documented source bonus. Before a live reply, `get_post` refreshes the
target and thread root; `replies` provides limited thread context to the LLM.
The account's following list prevents redundant FOLLOW actions.
A round can select a text action and a non-text interaction, but the interaction
slot is rate-limited: by default one LIKE/REPOST/FOLLOW/MENTION is eligible per
three successful CREATE_POST/REPLY actions. Configure it with
`CTN_TEXT_ACTIONS_PER_ENGAGEMENT`; dry-run history does not affect live cadence.

The default v0.3 policy evaluates strategic urgency and per-candidate
opportunity, then uses LinUCB + Softmax after five final CREATE_POST/REPLY
rewards when `CTN_ENABLE_LEARNING=true`. The selected feature vector and model
state are persisted for restart. Before that, Softmax samples the
heuristic/opportunity prior by action type and then target. LIKE, REPOST, FOLLOW,
MENTION and WAIT are not trained without action-specific outcomes. `--policy
heuristic` uses the heuristic Softmax branch and does not use LinUCB. Dry-run
actions never train the model. See `docs/technical_design_v0.3_learning.md`.

See `docs/technical_design_v0.2.md` for the full runtime and verification record.
