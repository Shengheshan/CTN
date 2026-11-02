"""Offline descriptive analysis of CTN action traces and counter observations."""

import argparse
import csv
import json
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from agent.content_features import extract_content_features
from config import BASE_DIR


METRICS = ("view_count", "like_count", "reply_count", "repost_count")
CURVE_BUCKETS = ("0-5m", "5-15m", "15-30m", "30-60m", "1-3h", "3-6h", "6-24h", ">24h")
COLORS = ("#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c")


def _parse_time(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    except (ValueError, TypeError):
        return None


def _age_bucket(seconds):
    for upper, label in ((300, "0-5m"), (900, "5-15m"), (1800, "15-30m"),
                         (3600, "30-60m"), (10800, "1-3h"), (21600, "3-6h"),
                         (86400, "6-24h")):
        if seconds <= upper:
            return label
    return ">24h"


def _crosses_bucket(start_seconds, end_seconds):
    return _age_bucket(start_seconds) != _age_bucket(end_seconds) and not any(
        abs(start_seconds - boundary) < 1.0
        for boundary in (300, 900, 1800, 3600, 10800, 21600, 86400))


def _write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _median(values):
    return statistics.median(values) if values else None


def _mean(values):
    return sum(values) / len(values) if values else None


def load_feedback(db_path: Path):
    uri = "file:" + str(db_path.resolve()).replace("\\", "/") + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        db.row_factory = sqlite3.Row
        actions = db.execute("""SELECT action_id,action_type,target_id,status,
                created_at,executed_at,remote_id,parameters_json,result_json,error_message
            FROM agent_actions WHERE action_type IN ('CREATE_POST','REPLY')
              AND remote_id IS NOT NULL AND executed_at IS NOT NULL
            ORDER BY executed_at,action_id""").fetchall()
        action_meta = {}
        for row in actions:
            params = json.loads(row["parameters_json"] or "{}")
            result = json.loads(row["result_json"] or "{}")
            content = (params.get("generated_content") or params.get("llm_draft")
                       or result.get("body") or "")
            features = params.get("content_features") or extract_content_features(content)
            posted = _parse_time(row["executed_at"])
            action_meta[row["action_id"]] = {
                "action_id": row["action_id"], "action_type": row["action_type"],
                "target_id": row["target_id"], "status": row["status"],
                "remote_id": row["remote_id"], "created_at": row["created_at"],
                "executed_at": row["executed_at"], "posted_hour_utc": posted.hour if posted else None,
                "tone": params.get("content_brief", {}).get("tone", "unknown"),
                "content": content, "content_features": features,
                "decision_trace": params.get("decision_trace"),
                "parameters": params, "result": result,
                "error_message": row["error_message"],
            }

        tables = {row["name"] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        scheduled = defaultdict(list)
        if "feedback_window_observations" in tables:
            rows = db.execute("""SELECT action_id,remote_id,window_label,
                    target_age_seconds,observed_at,actual_age_seconds,status,
                    metrics_json,delta_json,interval_start_at,
                    interval_start_age_seconds,crosses_age_window,error_type
                FROM feedback_window_observations ORDER BY action_id,observed_at""").fetchall()
            for row in rows:
                scheduled[row["action_id"]].append({
                    "action_id": row["action_id"], "remote_id": row["remote_id"],
                    "window_label": row["window_label"],
                    "target_age_seconds": row["target_age_seconds"],
                    "observed_at": row["observed_at"],
                    "age_seconds": row["actual_age_seconds"],
                    "status": row["status"],
                    "metrics": json.loads(row["metrics_json"] or "{}"),
                    "delta": json.loads(row["delta_json"] or "{}"),
                    "interval_start_at": row["interval_start_at"],
                    "interval_start_age_seconds": row["interval_start_age_seconds"],
                    "crosses_age_window": row["crosses_age_window"],
                    "error_type": row["error_type"], "sample_source": "scheduled_window",
                })

        legacy = defaultdict(list)
        if "observed_post_counters" in tables:
            rows = db.execute("""SELECT action_id,remote_id,observed_at,
                    metrics_json,delta_json,error_type
                FROM observed_post_counters ORDER BY action_id,observed_at,id""").fetchall()
            previous = {}
            for row in rows:
                action_id = row["action_id"]
                meta = action_meta.get(action_id)
                posted = _parse_time(meta["executed_at"]) if meta else None
                observed = _parse_time(row["observed_at"])
                if not posted or not observed:
                    continue
                age = max(0.0, (observed - posted).total_seconds())
                prior = previous.get(action_id)
                metrics = json.loads(row["metrics_json"] or "{}")
                delta = json.loads(row["delta_json"] or "{}")
                if not delta and prior and not row["error_type"]:
                    delta = {key: value - prior["metrics"].get(key, value)
                             for key, value in metrics.items()
                             if key in prior["metrics"]}
                start_age = prior["age_seconds"] if prior else None
                legacy[action_id].append({
                    "action_id": action_id, "remote_id": row["remote_id"],
                    "window_label": _age_bucket(age), "target_age_seconds": None,
                    "observed_at": row["observed_at"], "age_seconds": age,
                    "status": "ERROR" if row["error_type"] else "OBSERVED",
                    "metrics": metrics, "delta": delta,
                    "interval_start_at": prior["observed_at"] if prior else None,
                    "interval_start_age_seconds": start_age,
                    "crosses_age_window": (_crosses_bucket(start_age, age)
                                            if start_age is not None else None),
                    "error_type": row["error_type"], "sample_source": "continuous_monitor",
                })
                if not row["error_type"]:
                    previous[action_id] = {"observed_at": row["observed_at"],
                                           "age_seconds": age, "metrics": metrics}

    samples = []
    for action_id, meta in action_meta.items():
        window_rows = scheduled.get(action_id, [])
        if any(row["status"] in ("OBSERVED", "MISSED") for row in window_rows):
            chosen_rows = window_rows
        else:
            chosen_rows = legacy.get(action_id, [])
        for row in chosen_rows:
            enriched = {**row, **{
                "action_type": meta["action_type"], "tone": meta["tone"],
                "posted_hour_utc": meta["posted_hour_utc"],
                "primary_structure": meta["content_features"].get("primary_structure", "unknown"),
                "content_features": meta["content_features"],
            }}
            samples.append(enriched)
    return action_meta, samples


def build_window_curves(action_meta: dict, samples: list[dict]) -> list[dict]:
    levels = {}
    deltas = defaultdict(lambda: {"sum": 0.0, "posts": set(), "crossing_posts": set()})
    for sample in samples:
        action_id = sample["action_id"]
        if sample["status"] != "OBSERVED":
            continue
        bucket = _age_bucket(float(sample["age_seconds"]))
        if sample["metrics"]:
            key = (action_id, bucket)
            previous = levels.get(key)
            if previous is None or sample["age_seconds"] > previous["age_seconds"]:
                levels[key] = sample
        start_age = sample.get("interval_start_age_seconds")
        if start_age is None:
            continue
        crossed = sample.get("crosses_age_window")
        if crossed is None:
            crossed = _crosses_bucket(float(start_age), float(sample["age_seconds"]))
        for metric, change in sample["delta"].items():
            dkey = (action_id, bucket, metric)
            if crossed:
                deltas[dkey]["crossing_posts"].add(action_id)
            else:
                deltas[dkey]["sum"] += float(change)
                deltas[dkey]["posts"].add(action_id)

    grouped_levels = defaultdict(list)
    for (action_id, bucket), sample in levels.items():
        meta = action_meta[action_id]
        for metric in METRICS:
            value = sample["metrics"].get(metric)
            if value is None:
                continue
            group_key = (meta["action_type"], meta["tone"],
                         meta["content_features"].get("primary_structure", "unknown"),
                         meta["posted_hour_utc"], bucket, metric)
            grouped_levels[group_key].append((action_id, float(value)))

    output = []
    for key, items in sorted(grouped_levels.items(), key=lambda pair: str(pair[0])):
        action_type, tone, structure, posted_hour, bucket, metric = key
        action_ids = {action_id for action_id, _ in items}
        values = [value for _, value in items]
        change_stats = [deltas[(action_id, bucket, metric)] for action_id in action_ids]
        observed_changes = [item["sum"] for item in change_stats if item["posts"]]
        crossing_posts = {action_id for action_id, item in
                          ((aid, deltas[(aid, bucket, metric)]) for aid in action_ids)
                          if item["crossing_posts"]}
        output.append({
            "action_type": action_type, "tone": tone,
            "primary_structure": structure, "posted_hour_utc": posted_hour,
            "age_window": bucket, "metric": metric,
            "posts_with_observation": len(action_ids),
            "mean_cumulative_count": round(_mean(values), 3),
            "median_cumulative_count": _median(values),
            "posts_with_within_window_delta": len(observed_changes),
            "mean_observed_delta_within_window": round(_mean(observed_changes), 3)
            if observed_changes else None,
            "sum_observed_delta_within_window": round(sum(observed_changes), 3),
            "posts_with_cross_window_interval": len(crossing_posts),
            "evidence_type": "observational_association_only",
        })
    return output


def build_exposure_strata(action_meta: dict, samples: list[dict]) -> list[dict]:
    """Summarize engagement per 100 observed views within age/time/exposure strata."""
    latest = {}
    for sample in samples:
        if sample["status"] != "OBSERVED":
            continue
        metrics = sample["metrics"]
        views = metrics.get("view_count")
        if not isinstance(views, (int, float)) or views < 0:
            continue
        bucket = _age_bucket(float(sample["age_seconds"]))
        key = (sample["action_id"], bucket)
        if key not in latest or sample["age_seconds"] > latest[key]["age_seconds"]:
            latest[key] = sample

    def exposure_band(views):
        if views == 0:
            return "0"
        if views <= 5:
            return "1-5"
        if views <= 20:
            return "6-20"
        if views <= 100:
            return "21-100"
        return ">100"

    grouped = defaultdict(lambda: {"rates": [], "views": [], "actions": set()})
    for (action_id, age_bucket), sample in latest.items():
        meta = action_meta[action_id]
        views = float(sample["metrics"].get("view_count", 0))
        band = exposure_band(views)
        for metric in ("like_count", "reply_count", "repost_count"):
            count = sample["metrics"].get(metric)
            if count is None or views <= 0:
                continue
            key = (meta["action_type"], meta["tone"],
                   meta["content_features"].get("primary_structure", "unknown"),
                   meta["posted_hour_utc"], age_bucket, band, metric)
            group = grouped[key]
            group["rates"].append(float(count) / views * 100.0)
            group["views"].append(views)
            group["actions"].add(action_id)

    output = []
    for key, values in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        action_type, tone, structure, posted_hour, age_bucket, band, metric = key
        output.append({
            "action_type": action_type, "tone": tone,
            "primary_structure": structure, "posted_hour_utc": posted_hour,
            "age_window": age_bucket, "exposure_band_views": band,
            "metric": metric, "posts_with_observation": len(values["actions"]),
            "mean_interactions_per_100_views": round(_mean(values["rates"]), 4),
            "median_interactions_per_100_views": round(_median(values["rates"]), 4),
            "mean_cumulative_views": round(_mean(values["views"]), 3),
            "evidence_type": "exposure_stratified_observational_association_only",
        })
    return output

def build_timing_summary(curves: list[dict]) -> list[dict]:
    selected_windows = ("0-5m", "15-30m", "30-60m", "1-3h", "6-24h")
    return [row for row in curves if row["age_window"] in selected_windows]


def write_decision_traces(out_dir: Path, action_meta: dict) -> int:
    path = out_dir / "decision_trace.jsonl"
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for action_id, meta in action_meta.items():
            params = meta["parameters"]
            trace = meta["decision_trace"]
            complete = isinstance(trace, dict) and trace.get("version") == "decision-trace-v1"
            record = {
                "action_id": action_id, "action_type": meta["action_type"],
                "status": meta["status"], "target_id": meta["target_id"],
                "remote_id": meta["remote_id"], "executed_at": meta["executed_at"],
                "trace_complete": complete,
                "decision_trace": trace if complete else {
                    "selection_method": params.get("selection_method"),
                    "selection_score_components": params.get("score_components"),
                    "policy_features": params.get("policy_features"),
                    "legacy_trace_note": "Candidate alternatives were not persisted for this action.",
                },
                "content_features": meta["content_features"],
                "generated_content": meta["content"],
                "action_result": meta["result"],
                "error_message": meta["error_message"],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_overall_curves(action_meta: dict, samples: list[dict]) -> list[dict]:
    """Build chart series from post-level readings, not subgroup summaries."""
    latest = {}
    for sample in samples:
        if sample["status"] != "OBSERVED" or not sample["metrics"]:
            continue
        bucket = _age_bucket(float(sample["age_seconds"]))
        key = (sample["action_id"], bucket)
        if key not in latest or sample["age_seconds"] > latest[key]["age_seconds"]:
            latest[key] = sample
    grouped = defaultdict(list)
    for (action_id, bucket), sample in latest.items():
        action_type = action_meta[action_id]["action_type"]
        for metric in METRICS:
            value = sample["metrics"].get(metric)
            if value is not None:
                grouped[(action_type, bucket, metric)].append(float(value))
    return [{
        "action_type": action_type, "age_window": bucket, "metric": metric,
        "posts_with_observation": len(values),
        "median_cumulative_count": _median(values),
        "mean_cumulative_count": _mean(values),
    } for (action_type, bucket, metric), values in grouped.items()]

def write_curve_svg(path: Path, curves: list[dict]):
    metrics = METRICS
    action_types = sorted({row["action_type"] for row in curves})
    colors = {name: COLORS[index % len(COLORS)] for index, name in enumerate(action_types)}
    width, height, left, right, top, panel_height, gap = 1000, 920, 95, 30, 50, 185, 30
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#222}.small{font-size:12px}.title{font-size:20px;font-weight:bold}</style>',
            '<text x="30" y="28" class="title">Observed interaction curves (descriptive, not causal)</text>']
    xlabels = CURVE_BUCKETS
    usable_width = width - left - right
    for panel_index, metric in enumerate(metrics):
        y0 = top + panel_index * (panel_height + gap)
        panel_rows = [row for row in curves if row["metric"] == metric]
        max_value = max((row["median_cumulative_count"] or 0 for row in panel_rows), default=0)
        max_value = max(1.0, max_value)
        body.append(f'<text x="30" y="{y0 + 18}" font-size="15" font-weight="bold">{metric}</text>')
        for grid in range(4):
            y = y0 + 35 + (panel_height - 52) * grid / 3
            body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#ddd"/>')
            label = max_value * (1 - grid / 3)
            body.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="small">{label:.1f}</text>')
        for label_index, label in enumerate(xlabels):
            x = left + usable_width * label_index / max(1, len(xlabels) - 1)
            body.append(f'<text x="{x:.1f}" y="{y0+panel_height-8}" text-anchor="middle" class="small">{label}</text>')
        for action_type in action_types:
            values = {}
            for row in panel_rows:
                if row["action_type"] == action_type and row["median_cumulative_count"] is not None:
                    values[row["age_window"]] = row["median_cumulative_count"]
            points = []
            for label_index, label in enumerate(xlabels):
                if label not in values:
                    continue
                x = left + usable_width * label_index / max(1, len(xlabels) - 1)
                y = y0 + 35 + (panel_height - 52) * (1 - values[label] / max_value)
                points.append((x, y))
                body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{colors[action_type]}"/>')
            if len(points) > 1:
                coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
                body.append(f'<polyline points="{coords}" fill="none" stroke="{colors[action_type]}" stroke-width="2"/>')
    legend_x = left
    for action_type in action_types:
        body.append(f'<rect x="{legend_x}" y="{height-20}" width="12" height="12" fill="{colors[action_type]}"/>')
        body.append(f'<text x="{legend_x+17}" y="{height-10}" class="small">{action_type}</text>')
        legend_x += 150
    body.append('</svg>')
    path.write_text("\n".join(body), encoding="utf-8")


def run(db_path: Path, output_root: Path) -> Path:
    action_meta, samples = load_feedback(db_path)
    curves = build_window_curves(action_meta, samples)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    out_dir = output_root / f"feedback_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_fields = ["action_id", "action_type", "remote_id", "status", "sample_source",
                     "observed_at", "age_seconds", "window_label", "target_age_seconds",
                     "interval_start_at", "interval_start_age_seconds", "crosses_age_window",
                     "view_count", "like_count", "reply_count", "repost_count",
                     "delta_view_count", "delta_like_count", "delta_reply_count",
                     "delta_repost_count", "error_type", "tone", "primary_structure",
                     "posted_hour_utc", "content_features"]
    flat_samples = []
    for sample in samples:
        flat_samples.append({
            **{key: sample.get(key) for key in sample_fields},
            **sample["metrics"],
            **{f"delta_{key}": value for key, value in sample["delta"].items()},
            "content_features": json.dumps(sample["content_features"], ensure_ascii=False),
        })
    _write_csv(out_dir / "window_observations.csv", flat_samples, sample_fields)

    curve_fields = ["action_type", "tone", "primary_structure", "posted_hour_utc",
                    "age_window", "metric", "posts_with_observation",
                    "mean_cumulative_count", "median_cumulative_count",
                    "posts_with_within_window_delta", "mean_observed_delta_within_window",
                    "sum_observed_delta_within_window", "posts_with_cross_window_interval",
                    "evidence_type"]
    _write_csv(out_dir / "interaction_curves.csv", curves, curve_fields)
    _write_csv(out_dir / "posting_time_content_summary.csv", build_timing_summary(curves),
               curve_fields)
    exposure_rows = build_exposure_strata(action_meta, samples)
    _write_csv(out_dir / "exposure_stratified_interactions.csv", exposure_rows,
               ["action_type", "tone", "primary_structure", "posted_hour_utc",
                "age_window", "exposure_band_views", "metric", "posts_with_observation",
                "mean_interactions_per_100_views", "median_interactions_per_100_views",
                "mean_cumulative_views", "evidence_type"])
    trace_count = write_decision_traces(out_dir, action_meta)
    overall_curves = build_overall_curves(action_meta, samples)
    write_curve_svg(out_dir / "interaction_curves.svg", overall_curves)

    notes = """# Feedback analysis notes

These are observed counter changes and descriptive associations only. They are **not causal-effect estimates**.

- Posting time, action type, tone, and surface structure were not randomly assigned. Audience, topic, trend rank, concurrent posts, and platform exposure can confound comparisons.
- Counter deltas are changes between two successful reads. Their exact event time within that interval is unknown. Intervals crossing an age-window boundary are flagged and excluded from within-window delta totals rather than split or imputed.
- The first counter read for a post has no baseline delta. Missing or late windows remain missing; they are not interpolated.
- `view_count` is the API's aggregate view counter, not a verified unique-person or NPC exposure count. Counts do not establish persuasion or causal impact.
- `exposure_stratified_interactions.csv` reports crude interactions per 100 aggregate views, stratified by observed age window, posting hour, and fixed view-count band. It is a descriptive exposure adjustment; views are not randomized and can themselves reflect content and audience, so this is not a causal effect or a clean estimate of intrinsic content quality.
- The SVG uses post-level medians by action type and age window; sample size can vary across windows, so points do not necessarily represent the same post cohort. Detailed structure/hour strata remain in the CSV files.
- `primary_structure` is a deterministic surface-text grouping (question, link-bearing, first-person, emoji-bearing, exclamatory, multi-sentence, plain statement), not a semantic quality label.
- `decision_trace.jsonl` includes full candidate scores/features for actions recorded with `decision-trace-v1`. Earlier actions are labeled incomplete because alternative candidates were not stored at decision time.
- Reward changes and engagement deltas are separate: the former are model reward proxies; this report's curves use raw observed counters.
"""
    (out_dir / "analysis_notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps({
        "output_directory": str(out_dir),
        "posts_or_replies": len(action_meta),
        "counter_observations": len(samples),
        "curve_rows": len(curves),
        "decision_traces_exported": trace_count,
        "interpretation": "observational association only; no causal effects estimated",
    }, ensure_ascii=False))
    return out_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Offline CTN interaction-window analysis")
    parser.add_argument("--db", type=Path, default=BASE_DIR / "data" / "state.db")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "data" / "analysis")
    args = parser.parse_args(argv)
    run(args.db, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
