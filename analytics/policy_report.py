"""Offline reports for policy quality, execution and feedback completeness."""
import argparse
import csv
import json
import sqlite3
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from config import BASE_DIR


PANELS = (
    "Reward and 10-sample moving mean", "Execution success rate by action",
    "Action counts by day", "Observed interactions by feedback window",
    "Novelty score distribution", "Source validation status",
    "Decision score versus quality reward", "Policy parameter updates",
    "Feedback status and missingness",
)


def _connect(path):
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    db = sqlite3.connect(uri, uri=True)
    db.row_factory = sqlite3.Row
    return db


def _tables(db):
    return {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _rows(db, query):
    try:
        return [dict(row) for row in db.execute(query).fetchall()]
    except sqlite3.OperationalError:
        return []


def _csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def _panel(title, pairs):
    return {"title": title, "points": [{"label": str(k), "value": float(v)}
                                        for k, v in pairs]}


def _report_data(db_path: Path):
    with _connect(db_path) as db:
        tables = _tables(db)
        reward_columns = ({row[1] for row in db.execute("PRAGMA table_info(action_rewards)")}
                          if "action_rewards" in tables else set())
        reward_extra = [name for name in ("context_relevance_component",
                         "observed_feedback_component") if name in reward_columns]
        reward_fields = ["action_id","reward_version","feedback_window","execution_component",
                         "factual_component","novelty_component","compliance_component",
                         *reward_extra,"reward_total","reward_status","computed_at","calculation_json"]
        actions = _rows(db, """SELECT action_id,action_type,target_id,status,created_at,executed_at,
            score,parameters_json FROM agent_actions ORDER BY created_at""")
        rewards = (_rows(db, f"SELECT {','.join(reward_fields)} FROM action_rewards ORDER BY computed_at"
                         ) if "action_rewards" in tables else [])
        feedback = (_rows(db, """SELECT * FROM action_feedback_snapshots ORDER BY scheduled_at""")
            if "action_feedback_snapshots" in tables else [])
        assessments = (_rows(db, "SELECT action_id,features_json FROM content_assessments")
                       if "content_assessments" in tables else [])
        updates = (_rows(db, """SELECT action_id,policy_version,update_reason,learning_rate,
            created_at,update_applied,parameters_before_json,parameters_after_json
            FROM policy_updates ORDER BY created_at""") if "policy_updates" in tables else [])
        candidates = (_rows(db, """SELECT d.decision_id,d.created_at,c.action_type,c.base_score,
            c.action_prior,c.final_score,c.selected FROM policy_decisions d JOIN candidate_scores c
            USING(decision_id) ORDER BY d.created_at""") if "candidate_scores" in tables else [])

    action_counts=Counter(a["action_type"] for a in actions)
    action_success=defaultdict(lambda:[0,0])
    by_day=defaultdict(Counter)
    execution_attempt_count=0
    for action in actions:
        slot=action_success[action["action_type"]]
        if action["status"] != "DRY_RUN":
            execution_attempt_count+=1
            slot[1]+=1
            succeeded=(action["status"]=="SUCCEEDED" or
                       (action["action_type"]=="WAIT" and action["status"]=="WAIT"))
            if succeeded: slot[0]+=1
        day=(action["executed_at"] or action["created_at"] or "unknown")[:10]
        by_day[day][action["action_type"]]+=1
    success_points=[(key, ok/total if total else 0) for key,(ok,total) in sorted(action_success.items())]
    reward_points=[]
    reward_values=[]
    for reward in rewards:
        if reward["reward_total"] is not None:
            reward_values.append(float(reward["reward_total"]))
            reward_points.append((reward["computed_at"],float(reward["reward_total"])))
    rolling=[]
    for index,(_,value) in enumerate(reward_points):
        rolling.append((reward_points[index][0],statistics.mean(
            item[1] for item in reward_points[max(0,index-9):index+1])))
    interactions=defaultdict(list)
    missing=Counter()
    statuses=Counter()
    for item in feedback:
        statuses[item["feedback_status"]]+=1
        for metric in ("likes","replies","reposts","views","follows"):
            value=item.get(metric)
            if value is None: missing[metric]+=1
            else: interactions[(item["window_name"],metric)].append(float(value))
    interaction_points=[(f"{window}:{metric}",statistics.mean(values))
                       for (window,metric),values in sorted(interactions.items())]
    novelty_values=[]; source_status=Counter()
    for row in assessments:
        try: feature=json.loads(row["features_json"])
        except (TypeError,json.JSONDecodeError): continue
        if feature.get("novelty_score") is not None:
            novelty_values.append(float(feature["novelty_score"]))
        source_status[feature.get("source_validation",{}).get("validation_status","UNKNOWN")]+=1
    novelty_bins=Counter(min(9,int(value*10)) for value in novelty_values)
    day_points=[(day,sum(counts.values())) for day,counts in sorted(by_day.items())]
    updates_points=[("applied",sum(int(row["update_applied"]) for row in updates)),
                    ("not_applied",sum(1-int(row["update_applied"]) for row in updates))]
    missing_points=sorted(statuses.items()) + [(f"missing:{key}",value)
                                               for key,value in sorted(missing.items())]
    # Per-decision score/reward visualization is a correlation summary, not a causal fit.
    reward_by_action={row["action_id"]:row["reward_total"] for row in rewards
                      if row["reward_total"] is not None}
    score_reward=[]
    for action in actions:
        reward=reward_by_action.get(action["action_id"])
        if reward is not None and action["score"] is not None:
            score_reward.append((round(float(action["score"]),1),float(reward)))
    score_bins=defaultdict(list)
    for score,value in score_reward: score_bins[score].append(value)
    score_points=[(str(score),statistics.mean(values))
                  for score,values in sorted(score_bins.items())]
    exposure_groups=defaultdict(list)
    feature_by_action={}
    for row in assessments:
        try: feature_by_action[row["action_id"]]=json.loads(row["features_json"])
        except (TypeError,json.JSONDecodeError): pass
    action_by_id={row["action_id"]:row for row in actions}
    for sample in feedback:
        action=action_by_id.get(sample["action_id"])
        views=sample.get("views")
        if (not action or action["action_type"] not in ("CREATE_POST","REPLY")
                or views is None or views <= 0):
            continue
        band=("1-5" if views<=5 else "6-20" if views<=20 else
              "21-100" if views<=100 else ">100")
        try: posted_hour=datetime.fromisoformat((action.get("executed_at") or "").replace("Z","+00:00")).hour
        except ValueError: posted_hour=None
        structure=feature_by_action.get(action["action_id"],{}).get("primary_structure","unknown")
        for metric,column in (("likes","likes"),("replies","replies"),("reposts","reposts")):
            value=sample.get(column)
            if value is None: continue
            group=(action["action_type"],structure,posted_hour,sample["window_name"],band,metric)
            exposure_groups[group].append((float(value)/float(views))*100.0)
    exposure_rows=[]
    for key,values in sorted(exposure_groups.items(),key=lambda item:str(item[0])):
        action_type,structure,hour,window,band,metric=key
        exposure_rows.append({"action_type":action_type,"primary_structure":structure,
            "posted_hour_utc":hour,"feedback_window":window,"exposure_band_views":band,
            "metric":metric,"sample_size":len(values),
            "mean_interactions_per_100_views":round(statistics.mean(values),4),
            "median_interactions_per_100_views":round(statistics.median(values),4),
            "evidence_type":"descriptive_exposure_stratification_not_causal"})
    novelty_points=([(f"{i/10:.1f}-{(i+1)/10:.1f}",novelty_bins[i]) for i in range(10)]
                    if novelty_values else [])
    update_points=updates_points if updates else []
    panels=[
        _panel(PANELS[0],reward_points[-40:]),
        _panel(PANELS[1],success_points),
        _panel(PANELS[2],day_points[-30:]),
        _panel(PANELS[3],interaction_points[:30]),
        _panel(PANELS[4],novelty_points),
        _panel(PANELS[5],sorted(source_status.items())),
        _panel(PANELS[6],score_points),
        _panel(PANELS[7],update_points),
        _panel(PANELS[8],missing_points),
    ]
    ranges=[x for x in (a.get("executed_at") or a.get("created_at") for a in actions) if x]
    post_reply_total = action_counts.get("CREATE_POST", 0) + action_counts.get("REPLY", 0)
    summary={"data_range_start":min(ranges) if ranges else None,
      "data_range_end":max(ranges) if ranges else None,"action_count":len(actions),
      "action_type_counts":dict(action_counts),"execution_attempt_count":execution_attempt_count,
      "create_post_reply_selection_share": {
        "CREATE_POST": action_counts.get("CREATE_POST", 0) / post_reply_total if post_reply_total else None,
        "REPLY": action_counts.get("REPLY", 0) / post_reply_total if post_reply_total else None,
        "denominator": post_reply_total},
      "reward_count":len(rewards),
      "reward_version_counts":dict(Counter(row["reward_version"] for row in rewards)),
      "reward_status_counts":dict(Counter(row["reward_status"] for row in rewards)),
      "feedback_snapshot_count":len(feedback),"feedback_status_counts":dict(statuses),
      "feedback_missing_metric_counts":dict(missing),"novelty_assessed_count":len(novelty_values),
      "source_validation_counts":dict(source_status),"policy_update_count":len(updates),
      "policy_updates_applied":sum(int(row["update_applied"]) for row in updates),
      "candidate_score_count":len(candidates),
      "definitions":{"reward":"versioned normalized action quality and own-object feedback proxy",
        "success_rate":"SUCCEEDED/non-DRY_RUN actions divided by non-DRY_RUN attempts; WAIT is complete when status=WAIT",
        "interactions":"own-object and parent-thread counters by window; parent changes are not attributed to replies",
        "missing":"NULL counters are counted as missing and never converted to zero",
        "score_reward":"grouped action score and quality reward; descriptive only"}}
    return summary,panels,actions,rewards,feedback,assessments,updates,candidates,rolling,exposure_rows


def compare_priors(db_path: Path, configurations: dict):
    with _connect(db_path) as db:
        rows=_rows(db,"""SELECT d.decision_id,d.created_at,d.selection_method,
          d.decision_context_json,c.candidate_id,c.action_type,c.target_id,c.base_score,
          c.selected FROM policy_decisions d JOIN candidate_scores c USING(decision_id)
          ORDER BY d.created_at""")
    grouped=defaultdict(list)
    traces={}
    for row in rows:
        grouped[row["decision_id"]].append(row)
        if row["decision_id"] not in traces:
            try: traces[row["decision_id"]]=json.loads(row["decision_context_json"] or "{}")
            except (TypeError,json.JSONDecodeError): traces[row["decision_id"]]={}
    out=[]
    for decision_id,candidates in grouped.items():
        trace=traces[decision_id]
        baseline_row=next((row for row in candidates if row["selected"]),None)
        baseline=baseline_row["action_type"] if baseline_row else None
        method=trace.get("selection_method") or "unknown"
        scoring=trace.get("score_configuration") or {}
        current_priors=trace.get("action_priors") or {}
        trace_candidates={str(item.get("key")):item for item in trace.get("candidates",[])}
        for name,overrides in configurations.items():
            if not isinstance(overrides,dict):
                raise ValueError("prior configurations must be objects")
            if any(isinstance(value,bool) or not isinstance(value,(int,float)) or not 0<=value<=2
                   for value in overrides.values()):
                raise ValueError("offline Action Prior values must be numeric in [0,2]")
            priors={**current_priors,**overrides}
            ranked=[]
            for row in candidates:
                item=trace_candidates.get(str(row["candidate_id"]),{})
                action=row["action_type"]
                prior=float(priors.get(action,1.0))
                base=float(row["base_score"] or 0.0)
                adjusted=max(0.0,min(1.0,base*prior))
                opportunity=float((item.get("opportunity") or {}).get("score",0.0))
                if method in ("heuristic","prior_softmax"):
                    value=(float(scoring.get("wait_score",0.22)) if action=="WAIT" else
                        max(0.0,min(1.0,float(scoring.get("prior_heuristic",0.75))*adjusted+
                          float(scoring.get("prior_opportunity",0.20))*opportunity+
                          float(scoring.get("prior_urgency",0.05))*float((trace.get("strategic") or {}).get("urgency",0.0)))))
                elif action=="WAIT":
                    value=float(scoring.get("wait_score",0.22))
                elif action in ("LIKE","REPOST","FOLLOW","MENTION"):
                    value=adjusted
                else:
                    components=item.get("score_components") or {}
                    risk=sum(abs(float(component)) for key,component in components.items()
                             if key.endswith("penalty"))
                    value=(float(item.get("predicted_value") or 0.0)+
                           float(item.get("exploration_bonus") or 0.0)+(adjusted-base)+
                           float(components.get("action_priority",0.0))+
                           float(scoring.get("bandit_opportunity_prior",0.10))*opportunity-
                           float(scoring.get("bandit_risk_weight",1.0))*risk)
                ranked.append((value,row,item,prior))
            if not ranked: continue
            ranked.sort(key=lambda entry:(-entry[0],entry[1]["action_type"]!="WAIT",
                                          str(entry[1]["candidate_id"])))
            value,row,item,prior=ranked[0]
            out.append({"decision_id":decision_id,"created_at":row["created_at"],
              "selection_method":method,"configuration":name,
              "baseline_selected_action_type":baseline,
              "replayed_selected_action_type":row["action_type"],
              "candidate_id":row["candidate_id"],"replayed_score":round(value,6),
              "base_score":row["base_score"],"action_prior":prior,
              "same_as_baseline":row["action_type"]==baseline,
              "interpretation":"same recorded candidates and score formula; rankings only, no counterfactual outcome"})
    return out


def _write_png(data_path: Path, output_path: Path):
    script=Path(__file__).with_name("render_report_png.ps1")
    try:
        subprocess.run(["powershell.exe","-NoProfile","-ExecutionPolicy","Bypass",
                        "-File",str(script),"-DataPath",str(data_path),
                        "-OutputPath",str(output_path)],check=True,capture_output=True,text=True)
        return True
    except (OSError,subprocess.CalledProcessError):
        return False


def run(db_path: Path, output_root: Path, configurations: dict | None = None,
        source_provenance: str = "unspecified"):
    if source_provenance not in ("real", "test", "mock", "unspecified"):
        raise ValueError("source_provenance must be real, test, mock, or unspecified")
    summary,panels,actions,rewards,feedback,assessments,updates,candidates,rolling,exposure_rows=_report_data(db_path)
    summary["source_data_provenance"]=source_provenance
    summary["mock_data_used"]=source_provenance == "mock"
    summary["mock_rows_generated_by_report"]=False
    stamp=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    output=output_root/f"policy_report_{stamp}"
    output.mkdir(parents=True,exist_ok=True)
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    _csv(output/"actions.csv",actions); _csv(output/"rewards.csv",rewards)
    _csv(output/"feedback_snapshots.csv",feedback); _csv(output/"content_assessments.csv",[
      {"action_id":r["action_id"],**json.loads(r["features_json"])} for r in assessments])
    feature_map={r["action_id"]:json.loads(r["features_json"]) for r in assessments}
    reply_rows=[]
    for action in actions:
        if action["action_type"] != "REPLY": continue
        try: params=json.loads(action.get("parameters_json") or "{}")
        except (TypeError,json.JSONDecodeError): params={}
        feature=feature_map.get(action["action_id"],{})
        target=feature.get("target_features",{})
        timing=feature.get("timing_features",{})
        environment=feature.get("environment_features",{})
        for sample in feedback:
            if sample.get("action_id") != action["action_id"]: continue
            reply_rows.append({"action_id":action["action_id"],"parent_post_id":action.get("target_id"),
              "reply_text":(feature.get("content_features") or {}).get("content"),
              "parent_post_text":(params.get("context") or {}).get("post_body"),
              "window":sample.get("window_name"),"own_reply_delta_json":sample.get("delta_metrics_json"),
              "parent_post_delta_json":sample.get("parent_delta_metrics_json"),
              "parent_delta_attribution":"NOT_ATTRIBUTED_TO_REPLY",
              "context_relevance":feature.get("context_relevance_score"),
              "novelty":feature.get("novelty_score"),
              "parent_pre_action_metrics_json":target.get("pre_action_parent_metrics"),
              "topic_ids":target.get("topic_ids"),
              "parent_age_seconds":timing.get("parent_age_seconds"),
              "feed_sources":environment.get("feed_sources"),
              "interpretation":"descriptive association; no causal attribution"})
    _csv(output/"reply_context_feedback.csv",reply_rows)
    _csv(output/"policy_updates.csv",updates); _csv(output/"candidate_scores.csv",candidates)
    _csv(output/"reward_moving_average.csv",[{"computed_at":t,"moving_mean_10":v} for t,v in rolling])
    if configurations is not None:
        _csv(output/"prior_comparison.csv",compare_priors(db_path,configurations))
    _csv(output/"exposure_stratified_interactions.csv",exposure_rows)
    rewarded_ids={row["action_id"] for row in rewards if row.get("reward_total") is not None}
    panel_ns=[sum(1 for row in rewards if row.get("reward_total") is not None),summary["execution_attempt_count"],
              len(actions),sum(1 for row in feedback if row.get("views") is not None),
              summary["novelty_assessed_count"],len(assessments),
              sum(1 for row in actions if row["action_id"] in rewarded_ids),
              len(updates),len(feedback)]
    definitions=["Versioned quality plus action-owned feedback proxy; not causal",
      "Success rate: successes / real attempts; WAIT=complete; DRY_RUN excluded",
      "All logged action rows per UTC day",
      "Mean action-object counters by scheduled window; not unique-user exposure",
      "Word-shingle novelty 0-1; unknown scores omitted",
      "Source URL/alignment heuristic; not full fact-check",
      "Decision score vs later reward; descriptive only",
      "Policy parameter updates: applied vs not applied",
      "Feedback status; NULL metrics remain missing"]
    start_date=(summary["data_range_start"] or "none")[:10]
    end_date=(summary["data_range_end"] or "none")[:10]
    versions=",".join(sorted(summary["reward_version_counts"])) or "none"
    for index,panel in enumerate(panels):
        panel["subtitle"]=(f"n={panel_ns[index]} | {start_date} to {end_date} | Reward={versions}\n"
                            f"{definitions[index]} | NULLs remain missing")
    if rolling:
        panels[0]["line_points"]=[{"label":label,"value":value} for label,value in rolling[-40:]]
    chart_data=output/"dashboard_data.json"
    chart_data.write_text(json.dumps({"panels":panels},ensure_ascii=False),encoding="utf-8")
    png_ok=_write_png(chart_data,output/"policy_dashboard.png")
    summary["png_generated"]=png_ok
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    notes=("# Policy evaluation report\n\n"
      f"Data range: {summary['data_range_start'] or 'none'} to {summary['data_range_end'] or 'none'}\n\n"
      f"Actions: {summary['action_count']}; rewards: {summary['reward_count']}; "
      f"feedback snapshots: {summary['feedback_snapshot_count']}. Source provenance: {source_provenance}. "
      "The report generated no synthetic observations.\n\n"
      f"Reward versions: {json.dumps(summary['reward_version_counts'],ensure_ascii=False)}.\n\n"
      "Rewards are normalized quality and action-owned feedback proxies, not official game points, "
      "persuasion measures, or causal effects. Parent-post counter changes are separate and are not "
      "credited to replies. NULL metric values remain missing. "
      "No causal adjustment is claimed. A single action can have multiple versioned reward rows.\n\n"
      "When data are absent, the corresponding chart panel says Data insufficient. Novelty is a "
      "word-shingle similarity proxy. Source alignment is lexical and capped; it is not factual verification. "
      f"PNG generated: {png_ok}.\n")
    (output/"report_notes.md").write_text(notes,encoding="utf-8")
    print(json.dumps({"output_directory":str(output),"summary":summary},ensure_ascii=False))
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description="Offline CTN v0.3 policy and feedback report")
    parser.add_argument("--db",type=Path,default=BASE_DIR/"data"/"state.db")
    parser.add_argument("--out",type=Path,default=BASE_DIR/"data"/"analysis")
    parser.add_argument("--compare-priors",type=Path,help="JSON object of prior configurations")
    parser.add_argument("--source-provenance",choices=("real","test","mock","unspecified"),default="unspecified")
    args=parser.parse_args(argv)
    configs=json.loads(args.compare_priors.read_text(encoding="utf-8")) if args.compare_priors else None
    run(args.db,args.out,configs,args.source_provenance)
    return 0

if __name__=="__main__": raise SystemExit(main())
