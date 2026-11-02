# Read-only action and counter monitor

Run from `D:\CTN-bot`:

```powershell
.\.venv\Scripts\python.exe -B monitor.py
.\.venv\Scripts\python.exe -B monitor.py --watch --interval 60
```

The monitor prints recent `agent_actions` as JSON lines and polls public
`social.counters()` for recent successful CREATE_POST and REPLY records. Every
sample is stored in `observed_post_counters` in `data/state.db`, with the UTC
observation time, available view/like/reply/repost counts, changes since the
previous successful sample, and an error type if a read failed. Ctrl+C stops
watch mode. `--limit` controls the maximum recent records inspected per pass.

This command never logs in, publishes, reports, or updates the policy model.
Counter values are observed platform engagement, not verified opinion changes,
official points, paid views, or causal impact. If there are no successful
published posts or replies in the action journal, only action records print.
