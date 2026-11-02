# CTN Agent architecture decisions

## ADR-001 Rebuild world state from snapshots

Raw and normalized observations remain in SQLite. WorldState and encoded
features are rebuilt for each snapshot so restarts and replays remain possible.

## ADR-002 Keep dynamic topics optional

Batch topic extraction and registry reuse remain available, but unstable
semantic quality must not stop the runtime or determine action eligibility.

## ADR-003 Separate observation, policy, execution and feedback

Observation describes the environment. Policy returns a structured ActionPlan.
The executor alone calls write APIs. AgentState and action history are stored
separately from WorldState in the same SQLite database.

## ADR-004 Use a deterministic heuristic before learning

Policy v0.2 has centralized, explainable weights and an explicit WAIT option.
At v0.2, contextual UCB and Softmax remained future interfaces until a
versioned reward and credit-assignment path existed. ADR-008 and ADR-009 record
the v0.3 quality-feedback path that now supplies that bounded training signal.
Scores remain estimates, not official game impact.

## ADR-005 Reserve before writes and fail closed on uncertainty

Each live action is reserved in SQLite before the SDK call. Definitive success
and failure are recorded separately from UNKNOWN responses. An UNKNOWN action
key blocks automatic repetition. Feed lock or unavailable lock state prevents
writes; 429 retries are bounded.

## ADR-006 Ground generated text in exact source excerpts

Content generation uses the supplied competition LLM to write a short,
conversational post or reply from observed text and attributed sources.
ActionPlan records the intended tone and timing constraints. Empty, overlong,
placeholder and unsupported numerical claims stop publication. The prompt
and checks reduce errors but do not guarantee factuality.

## ADR-007 Make strategic and opportunity uncertainty explicit

The v0.2 heuristic omitted global strategic context and treated unknown task
relevance as a neutral constant. v0.2 adds separate strategic and opportunity
assessments. Unavailable favorability, alignment and competition remain marked
missing; no poll position or causal effect is inferred from post heat.

## ADR-008 Keep contextual UCB and Softmax isolated from untrusted outcomes

The v0.2 engagement-proxy model remains available for audit and is not loaded.
The v0.3 contextual LinUCB is trained only from versioned final rewards for
CREATE_POST and REPLY, using their selected candidate feature vectors. Likes,
reposts, follows, mentions and WAIT have no action-specific outcome signal and
are not trained. Engagement counters are descriptive response signals, not
evidence of persuasion; parent-post changes are never credited to a reply.

## ADR-009 Add auditable v0.3 quality feedback with learning off by default

Every new decision stores its complete trace and all evaluated candidate score
components. Deterministic novelty and conservative source matching are persisted
with explicit UNKNOWN values. Scheduled action-specific feedback is stored in
append-only snapshots; missing values remain NULL. The `reward-v1-quality`
components are execution, source alignment, novelty, and immediate platform
acceptance. Engagement counters and political preference outcomes are excluded.

`CTN_ENABLE_LEARNING=false` is the code default. With learning disabled, final
rewards are logged with `update_applied=0`. When enabled, the runtime restores
and updates the versioned LinUCB from each eligible final CREATE_POST/REPLY
reward, and records before/after model state. A clipped action-prior learner
helps the first five cold-start updates; it is not trained from engagement-only
actions. A persistent learning-start timestamp excludes actions created before
online learning began; their existing rewards are not backfilled into the
learner. Known FAILED/BLOCKED text actions may contribute an execution-only
outcome; UNKNOWN outcomes remain invalid. Reward versions, component weights
and calculation evidence are retained. Reports remain descriptive and make no
causal claim. See `technical_design_v0.3.md` for details.

## ADR-010 Schedule post coverage separately from observed reward

The scorer exposes configurable action-priority terms for `CREATE_POST` and
`REPLY`. A rolling successful text-action window can enforce a configurable
minimum share of eligible original posts; the first action remains unconstrained
until there is history, and ineligible/locked/budget-exhausted rounds still use
the existing candidate and WAIT behavior. A previously handled news event can
become eligible again after its configurable cooldown, with a per-snapshot
idempotency key; exact duplicate-content checks remain active. Scheduling priority is recorded in
the decision trace and never inserted into `Observed Reward`.

Reply traces persist content, target/thread, timing, and environment features.
For each reply, windowed snapshots separately store counters for the reply
object and its parent post. The reply's own counters may contribute to its
observed feedback component; parent-post changes are retained for descriptive
analysis and are explicitly not credited to the reply. Missing API values stay
NULL. Comparisons by content, target context, or timing are observational and
do not establish causal effects. Reward v2 records action-specific quality and
own-object feedback; novelty remains an analysis/exploration feature and is
excluded from reward. No individual voter stance is used for targeting.

Optional action-prior learning clips each mean update to 0.01 by default,
decays the deviation from the neutral baseline by 2% per update, and caps each
action prior to [0.8, 1.2]. A single unusually high reward therefore cannot
cause a large immediate policy shift.

Non-LLM interactions share one rate limit: by default, LIKE/REPOST/FOLLOW/MENTION
are eligible at most once per three successful CREATE_POST/REPLY actions since
the previous interaction. The limit is configurable and does not enter reward.

## ADR-011 Normalize news links and contain text-generation failures

News API article URLs may be site-relative. After selecting a **news** source,
resolve its URL against the SDK news-site base (`https://sites.legitreal.com`)
and update the selected source in ActionPlan context. Use the same absolute URL
for the prompt, published citation, character budget, and source matching.
Both ordinary and trend-driven CREATE_POST actions require a citable news
article. A historical Trending social post is only a topic signal: match it
to news already collected in the current snapshot (up to the latest 20 by
default) before creating a trend-driven candidate; without matching news,
create no such candidate.

If the Trending post contains a news URL, match it exactly against currently
observed articles after normalizing site-relative paths. Otherwise remove
common words and require at least two post/article token overlaps, including
at least one title overlap. The article must itself be campaign relevant; a
linked URL without a current article match does not fall back to term matching.
Keep the social post ID as generation context and Decision Trace evidence.
Published text appends the matched news article's full original URL, never
the Trending post URL or a `Source: Legit Social` citation. Preserve already
published posts and raw/SQLite snapshots. Candidate matching and news-only
publication behavior passed offline regression tests; live platform link
preview remains unverified.

Strip model-written hashtags locally before validation and retain the
application's existing post-generation tag append. A 429 that has exhausted
the LLM client's three bounded attempts must not trigger Runtime's immediate
same-round full-generation retry; other retry behavior remains bounded.
For posts based on a critical source, distinguish the source's argument from
the account's response. Semantic review judges the finished post's argument:
adopting an opposing conclusion without a substantive Vote Yes response calls
for revision, while acknowledging a concern alone does not require a slogan.

Status: implemented; 35 related offline tests passed across policy, content,
LLM client, and existing repair paths. The content layer rejects hot-trend
plans without a news article before calling the LLM. No live write or link
preview was performed.


## ADR-012 Restore direct-discussion prompts and score hot-news cooling

The user selected the earlier direct-discussion REPLY and CREATE_POST writing
templates, without the later family Persona. Keep the existing JSON keys,
variable injection, tone/profile selection, dynamic body budget, source-link
postprocessing, semantic review and bounded retries. Both initial generation
and repair read the restored constraints. The previous templates are saved in
`docs/prompt_templates_before_direct_rollback_2026-11-02.json`. This is a
reversible experiment; sparse engagement counters and frequent LLM 429 failures
do not establish that the Persona caused the observed decline.

A historical Trending post must have an observed historical or current heat
score at or above configurable `CTN_HOT_NEWS_SCORE_THRESHOLD` (default 30)
and match a currently collected news article before it can generate a hot-news
candidate. Current Trending status gives no cooling; absent current heat gives
full cooling; other non-Trending observations receive at least 0.6 cooling.
The hot candidate receives a heat bonus of 0.14 times normalized current heat
and a cooling penalty of 0.28 times cooling. If no own-post reply defers this
layer, its entry probability is `0.5 * (1 - 0.8 * minimum_cooling)`,
falling to 0.1 if all candidates have full cooling. Within the layer,
selection is Softmax over heat-adjusted scores. The previous fixed 0.5 gate
with uniform candidate selection is superseded. Historical hot topics remain
eligible after cooling, but less likely to take a round.

The new terms affect Decision Score and selection probability, not Observed
Reward. Hot-news actions remain excluded from main-policy reward updates and
cite the matched news URL rather than the social post URL. No live platform
write was made for this change. The combined 88-test offline suite passed.


## ADR-013 Rotate relevant popular post tags without expanding tag-feed reads

Keep the existing two popular tag feeds read by Runtime. For CREATE_POST,
sample up to three current popular tags from a broader approved pool. Accept
general campaign tags and source-context tags for nurses, businesses, or
locations only when the selected candidate's news text supports them.
Reject opposing, unrelated, malformed, or absent tags. Weight the draw by
logarithmic popularity and discount tags used in the previous three successful
original posts. The content generator still reserves the source citation and
full article URL before finalizing tags; insufficient space removes tags from
the end. The ActionPlan records final tags. No change is made to Reward,
StateEncoder, or the SDK execution path. Offline relevant tests and a
three-tag generation probe passed; live publication was not performed.
