# CTN Agent learning policy v0.3

## Why the v0.2 score was insufficient

The v0.2 scorer used post heat, freshness, article evidence, default relevance
and a fixed recent-action penalty. It did not consume the global encoded vector
or update from feedback. With only dry-run history, its repeat penalty could
push all candidates below the WAIT threshold. These were engineering rules,
not learned estimates of action value.

## Decision path

`WorldState → StateEncoder → StrategicEvaluator → CandidateGenerator →
OpportunityEvaluator + PolicyScorer → LinUCB + Softmax → ActionPlan →
Executor → delayed feedback`.

`StrategicEvaluator` reports a provisional urgency from observed trending and
hot-velocity data. Favorability is `None`: there is no poll, side-alignment or
official score in WorldState. `OpportunityEvaluator` records observed influence,
freshness, reach and any externally supplied relevance; it lists missing
relevance, strategic fit and competition rather than inventing them. These
components are used both by the cold-start prior and the contextual model.

The context feature vector is versioned and fixed-length: intercept, 12 global
features, 8 post features, 1 event feature, action-type one-hot encoding, and
opportunity score, evidence coverage, urgency and risk. It excludes author
identity and untrusted topic labels.

For candidate feature vector x, the model predicts `theta^T x`, adds
`alpha * sqrt(x^T A_inverse x)`, a small opportunity prior, and subtracts
observed execution risk. Softmax converts those scores to probabilities at a
configurable temperature. WAIT remains a real option. The chosen plan stores its feature vector, selection method and score
components in the decision trace and `candidate_scores`. For each eligible final
reward, Runtime reloads that selected vector, updates LinUCB with the bounded
reward, and persists the model in `policy_states` under the feature version.

The default CLI policy is `bandit`. When `CTN_ENABLE_LEARNING=true`, final
CREATE_POST/REPLY rewards update LinUCB; at five eligible updates selection
uses LinUCB + Softmax. During cold start, Softmax samples a heuristic,
opportunity and urgency prior; the first five eligible outcomes also update a
small bounded action prior. Action type is sampled before target, avoiding a
bias toward action types with many candidates. `--policy heuristic` uses a
heuristic Softmax comparison and never uses LinUCB scores. LIKE, REPOST, FOLLOW
and MENTION retain heuristic scores because they lack action-specific outcomes.

## Feedback and limits

Only successful real CREATE_POST and REPLY actions with a remote ID can
receive scheduled own-object feedback. T+5m/T+30m/T+2h/T+6h windows are stored
with missing fields preserved as NULL; reply-object and parent-post changes
remain separate, and parent deltas are never attributed to the reply. The v0.3
reward combines available execution, source/content quality, compliance, and
own-object feedback components; its calculation and reward version are stored
in `action_rewards`.

When learning is enabled, final eligible CREATE_POST/REPLY rewards update
LinUCB using the selected candidate vector saved in `candidate_scores`. The
model is persisted in `policy_states`; an action can update it at most once.
Known FAILED/BLOCKED text actions receive an execution-only final reward and
may train as a negative outcome. UNKNOWN outcomes are invalid and do not train.
Dry-run actions, WAIT, LIKE, REPOST, FOLLOW, MENTION and independent hot-news
layer actions do not train. A persistent learning-start timestamp excludes
actions created before online learning began; old rewards are not backfilled.
These measurements describe observed response and content quality; they are
not causal proof of persuasion or an official competition score. The code
default for learning is off; this machine's `.env` opts in.

Text generation uses the configured draft temperature (default `0.9`), repair
temperature (default `0.35`) and review temperature (default `0.1`). A content
generation failure gets one bounded fresh-generation retry unless the failure
is structural. A platform `400`, `413` or `422` rejection gets one regenerated
submission. The LLM client serializes requests per team code and schedules them
at least 21 seconds apart across local processes, matching the current published
three-requests-per-minute quota. HTTP `429` extends that shared cooldown using
`Retry-After`; timeouts and server-side `5xx` responses remain `UNKNOWN` to
avoid duplicate posts or replies.

The policy remains limited by unknown favorability, task alignment,
competition and reward attribution. Future work can add verified official
outcome signals and compare learned policy performance with the heuristic
baseline before increasing live exploration.

## Live action and content constraints

The runtime admits only recent H2D2 referendum sources: posts within 36 hours
and news within 72 hours. A conservative lexical eligibility gate excludes
unrelated topics; it does not replace dynamic topic classification. Explicit
positive alignment permits LIKE, REPOST, FOLLOW and a limited MENTION on a
relevant post previously published by this agent. MENTION is capped at one
recipient per own post and one mention per recipient in action history.

In bandit mode, cold-start and trained selection both sample action type
before target with Softmax. This avoids choosing REPLY mainly because it has
more candidates. Live rounds also vary the requested interval by ±30%; the
60-second setting therefore means approximately 42 to 78 seconds between
rounds. `--policy heuristic` samples the heuristic scores with Softmax; it is a
comparison branch and does not use LinUCB.

CREATE_POST and REPLY use the competition LLM. The chosen plan stores tone,
purpose, observation time and source-age limit. CREATE_POST receives an
attributed news source; REPLY receives its target post without unverified
neighboring replies or unrelated news summaries. Output is checked for length,
placeholders, @mentions and numbers unsupported by the supplied text. A news
post includes its source title and URL. Generated text and score components
are stored in the action journal. These checks cannot prove every qualitative
claim true, so live output still requires review through the platform.

After local validation, the official LLM performs one independent semantic
review against the cited article or reply target. It checks that the draft
addresses the same concern, keeps a coherent Yes position, avoids unsupported
personal or factual claims, and reads conversationally. A rejected draft gets
the existing single repair attempt and must pass review again before execution;
the review decision and reason are stored with the action.

CREATE_POST now receives a topic-specific focus derived from the cited article.
The prompt separates the source's reported viewpoint from the account's own
Yes-side stance. A local gate rejects an explicit No vote, unrelated voter-ID
or ballot topics, and posts that fail to address a sunset/review concern when
that is the cited article's central issue. A failed post gets one LLM repair
attempt and is skipped if it still fails.

For REPLY, unrelated campaign news is no longer attached by default. The prompt
requires identifying and answering the author's specific question in the same
subtopic. A local semantic gate rejects obvious neighboring-topic shifts and
requires a first-person condition when the author asks what would change their
vote. It also rejects invented personal medical experiences, requires the
reply to retain a concrete topic anchor, and checks that the Vote Yes position
is stated clearly. When a post distinguishes free vaccination from venue entry
checks, the prompt requires a proportionate safeguard on those checks and the
gate rejects replies that make abolishing the central checks the condition for
voting Yes. A failed draft gets one LLM repair attempt; if it still fails, the
reply is skipped. The gate is intentionally conservative and does not claim to
prove persuasion quality.

Replies about sunset clauses, end dates, grace periods, or review windows use a
separate focus and must mention that timing safeguard, so they do not fall back
to generic vaccine-access language.
Staffing and shift-cover replies receive their own focus, and drafting meta
language (such as narrating the act of writing the reply) is rejected before
publication.
Rapid-test alternatives are separated from vaccine-protection duration; a
reply to a test-access post must address that testing option, and cannot invent
booster or dose schedules absent from the original post.

The runtime also reads `social.popular_tags()` once per round. CREATE_POST
may append up to two currently popular, allowlisted tags: one general H2D2 or
referendum tag and one Vote Yes aligned tag. The content generator checks the
character limit after appending them and records `selected_tags` for audit.

An authenticated round also reads `account.feed_home`, posts under those two
popular tags, unresolved notification posts, and the current following list.
They join the saved raw snapshot and normalized post table with separate feed
source labels. A notification, Home, or tag source contributes a small explicit
score bonus. A selected live REPLY refreshes the target through `get_post`,
reads a few replies for context, and uses the original thread root when
publishing. A failed refresh prevents that write. Reporting, deleting,
unfollowing, and profile edits have no verified automatic eligibility rule, so
they are outside the current action set.

## Verification

Run from `D:\CTN-bot`:

```powershell
.\.venv\Scripts\python.exe -B -m unittest -v test_policy test_action_policy test_v02 test_learning_policy
.\.venv\Scripts\python.exe -B main.py --mode dry-run --once --policy bandit
.\.venv\Scripts\python.exe -B main.py --mode dry-run --once --policy heuristic
```

Tests cover explicit missing signals, feature dimensions, LinUCB learning and
uncertainty reduction, stable Softmax probabilities, cold-start-to-bandit
switch, reward restrictions, delayed feedback, model persistence and restart
updates. No test or development command publishes platform content.
