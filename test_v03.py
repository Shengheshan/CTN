import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.executor import ActionExecutor, ExecutionResult
from agent.feedback import V03FeedbackSystem
from agent.history import ActionHistory
from agent.v03_store import V03Store
from policy.feature_analyzer import ContentFeatureAnalyzer
from policy.learning import ActionPriorLearningPolicy, HeuristicPolicy
from policy.models import ActionPlan, CandidateAction
from policy.novelty import novelty
from policy.reward import RewardEvaluator
from policy.bandit import FEATURE_DIM, FEATURE_VERSION
from policy.settings import PolicySettings


def test_config(windows=(300, 1800, 7200, 21600), enable_learning=False):
    return SimpleNamespace(enable_feedback=True, enable_learning=enable_learning,
        feedback_windows=windows, reward_version="reward-v1-quality", reward_weights={},
        novelty_history_size=100, novelty_duplicate_threshold=0.9)


def make_plan(action="CREATE_POST", text="A clear and original message about public health today.", context=None):
    candidate=CandidateAction(action, target_id=45 if action in ("REPLY", "LIKE", "REPOST", "FOLLOW", "MENTION") else None,
                              context=context or {})
    plan=ActionPlan.from_candidate(1,candidate,0.6,{"base":0.6})
    plan.generated_content=text if action in ("CREATE_POST","REPLY") else None
    plan.parameters["policy_features"]=[0.1,0.2]
    plan.parameters["decision_trace"]={"version":"decision-trace-v1","decision_id":"decision-1",
        "snapshot_id":1,"policy_version":"policy-test","encoder_version":"encoder-test",
        "selection_method":"heuristic","candidates":[{"key":candidate.key,"action_type":action,
        "target_id":None,"base_score":0.6,"action_prior":1.0,"heuristic_score":0.6,
        "selection_score":0.6,"selection_probability":1.0,"exploration_bonus":0.0,
        "selected":True,"score_components":{"cost_penalty":-0.1,"repeat_penalty":-0.02},
        "policy_features":[0.1,0.2]}]}
    return plan


def attach_valid_policy_features(plan):
    features = [1.0] + [0.0] * (FEATURE_DIM - 1)
    plan.parameters["policy_features"] = features
    plan.parameters["decision_trace"]["candidates"][0]["policy_features"] = features
    return plan


class FakeCounters:
    def __init__(self, response=None, error=None): self.response=response or {}; self.error=error
    def counters(self, remote_id):
        if self.error: raise self.error
        return self.response


class FakeHTTPError(Exception):
    def __init__(self,status,retry_after=None): self.status=status; self.retry_after=retry_after


class V03Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.db=Path(self.temp.name)/"state.db"
        self.history=ActionHistory(self.db)
        self.store=V03Store(self.db)

    def tearDown(self): self.temp.cleanup()

    def persist(self, plan, status="SUCCEEDED", remote="9001"):
        self.history.save_result(plan,status,remote_id=remote)
        return plan

    def test_01_candidate_scores_and_decision_trace_persist(self):
        plan=self.persist(make_plan())
        self.store.record_decision(plan)
        with self.store.connect() as db:
            decision=db.execute("SELECT candidate_count,selected_action_id FROM policy_decisions").fetchone()
            score=db.execute("SELECT base_score,action_prior,cost_penalty,risk_penalty,selected FROM candidate_scores").fetchone()
        self.assertEqual(decision[0],1); self.assertEqual(decision[1],plan.action_id)
        self.assertEqual(tuple(score),(0.6,1.0,0.1,0.02,1))

    def test_02_action_prior_configuration_loads(self):
        with patch.dict(os.environ,{"CTN_ACTION_PRIORS":'{"CREATE_POST":1.4,"REPLY":0.8}' }):
            settings=PolicySettings.from_env()
        self.assertEqual(settings.action_priors["CREATE_POST"],1.4)
        self.assertEqual(settings.action_priors["REPLY"],0.8)
        self.assertEqual(settings.action_priors["LIKE"],1.0)
        with patch.dict(os.environ,{"CTN_MINIMUM_POST_COVERAGE":"0.4",
                                   "CTN_POST_COVERAGE_WINDOW":"12",
                                   "CTN_CREATE_POST_PRIORITY":"0.2",
                                   "CTN_REPLY_PRIORITY":"-0.1"}):
            configured=PolicySettings.from_env()
        self.assertEqual(configured.minimum_post_coverage,0.4)
        self.assertEqual(configured.post_coverage_window,12)
        self.assertEqual(configured.create_post_priority,0.2)
        self.assertEqual(configured.reply_priority,-0.1)

    def test_03_exact_duplicate_is_detected(self):
        text="A clear and original message about public health today."
        result=novelty(text,[text])
        self.assertTrue(result["duplicate"]); self.assertEqual(result["novelty_score"],0)

    def test_04_different_text_is_not_duplicate(self):
        result=novelty("Fresh words discuss a different community concern today.",
                       ["The river cleanup volunteers meet at the eastern library tomorrow."])
        self.assertFalse(result["duplicate"])

    def test_05_invalid_news_url_is_unverified(self):
        plan=make_plan(text="A report is here https://not-saved.invalid/story")
        result=ContentFeatureAnalyzer().analyze(plan,[])
        self.assertEqual(result["source_validation"]["validation_status"],"UNVERIFIED")
        self.assertEqual(result["source_alignment_score"],0.0)

    def test_06_explicit_numeric_source_conflict_is_flagged(self):
        article={"url":"https://news.test/ward","title":"Ward has 20 nurses",
                 "summary":"The ward staffing report counts 20 nurses."}
        plan=make_plan(text="The ward needs 40 nurses. Source: https://news.test/ward",
                       context={"articles":[article]})
        result=ContentFeatureAnalyzer().analyze(plan,[])
        self.assertEqual(result["fact_check_status"],"CONTRADICTED")
        self.assertEqual(result["source_validation"]["validation_status"],"CONTRADICTED")

    def test_07_success_schedules_each_feedback_window(self):
        plan=self.persist(make_plan())
        count=self.store.schedule_feedback(plan,ExecutionResult("SUCCEEDED","9001"),(300,1800))
        self.assertEqual(count,2)
        with self.store.connect() as db:
            rows=db.execute("SELECT window_name,feedback_status FROM action_feedback_snapshots ORDER BY scheduled_at").fetchall()
        self.assertEqual([r[0] for r in rows],["t+300s","t+1800s"])
        self.assertEqual([r[1] for r in rows],["PENDING","PENDING"])

    def test_08_not_due_window_is_not_collected_early(self):
        plan=self.persist(make_plan())
        self.store.schedule_feedback(plan,ExecutionResult("SUCCEEDED","9001"),(300,))
        self.assertEqual(self.store.due_feedback(),[])

    def test_09_feedback_window_cannot_be_scheduled_twice(self):
        plan=self.persist(make_plan())
        self.store.schedule_feedback(plan,ExecutionResult("SUCCEEDED","9001"),(300,))
        self.store.schedule_feedback(plan,ExecutionResult("SUCCEEDED","9001"),(300,))
        with self.store.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM action_feedback_snapshots").fetchone()[0],1)

    def test_10_missing_metrics_remain_null(self):
        plan=self.persist(make_plan())
        self.store.schedule_feedback(plan,ExecutionResult("SUCCEEDED","9001"),(300,))
        with self.store.connect() as db:
            db.execute("UPDATE action_feedback_snapshots SET scheduled_at='2000-01-01T00:00:00+00:00'")
        item=self.store.due_feedback()[0]
        self.store.complete_feedback(item,{"like_count":2},"PARTIAL")
        with self.store.connect() as db:
            row=db.execute("SELECT likes,replies,views,delta_metrics_json FROM action_feedback_snapshots").fetchone()
        self.assertEqual(row[0],2); self.assertIsNone(row[1]); self.assertIsNone(row[2]); self.assertIsNone(row[3])

    def test_reply_feedback_keeps_reply_and_parent_deltas_separate(self):
        context={"source_post_id":45,"post_body":"A public health discussion"}
        plan=make_plan("REPLY",context=context)
        self.persist(plan)
        baseline=FakeCounters({"like_count":10,"reply_count":2,"repost_count":1,"view_count":100})
        system=V03FeedbackSystem(baseline,self.db,test_config(windows=(300,)))
        system.record_content(plan)
        system.record_result(plan,ExecutionResult("SUCCEEDED","9001"))
        with self.store.connect() as db:
            db.execute("UPDATE action_feedback_snapshots SET scheduled_at='2000-01-01T00:00:00+00:00'")
        changed=FakeCounters({"like_count":13,"reply_count":4,"repost_count":1,"view_count":120})
        system.social=changed
        system.process_due()
        with self.store.connect() as db:
            row=db.execute("""SELECT likes,delta_metrics_json,parent_remote_id,
              parent_delta_metrics_json,raw_metrics_json FROM action_feedback_snapshots""").fetchone()
        self.assertEqual(row[0],13)
        self.assertIsNone(row[1])
        self.assertEqual(row[2],"45")
        self.assertEqual(json.loads(row[3])["likes"],3)
        self.assertIn("parent_delta_is_not_attributed_to_reply",row[4])

    def test_hot_news_layer_feedback_is_logged_but_never_rewarded_or_learned(self):
        plan=make_plan()
        plan.parameters["exclude_from_reward"]=True
        plan.parameters["reward_exclusion_reason"]="independent_hot_news_layer"
        self.persist(plan)
        system=V03FeedbackSystem(
            FakeCounters({"like_count":3,"reply_count":1,"repost_count":0,"view_count":20}),
            self.db,test_config(windows=(300,),enable_learning=True))
        system.record_result(plan,ExecutionResult("SUCCEEDED","9001"))
        with self.store.connect() as db:
            db.execute("UPDATE action_feedback_snapshots SET scheduled_at='2000-01-01T00:00:00+00:00'")
        result=system.process_due()
        with self.store.connect() as db:
            feedback=db.execute("SELECT feedback_status,likes,replies FROM action_feedback_snapshots").fetchone()
            reward_count=db.execute("SELECT COUNT(*) FROM action_rewards WHERE action_id=?",
                                    (plan.action_id,)).fetchone()[0]
            update_count=db.execute("SELECT COUNT(*) FROM policy_updates WHERE action_id=?",
                                    (plan.action_id,)).fetchone()[0]
        self.assertEqual(tuple(feedback),("COLLECTED",3,1))
        self.assertEqual(reward_count,0)
        self.assertEqual(update_count,0)
        self.assertEqual(result["feedback_collected"],1)
        self.assertEqual(result["rewards_recomputed"],0)
        self.assertEqual(result["policy_updates"],0)

    def test_reward_uses_only_own_feedback_and_action_specific_context(self):
        result=RewardEvaluator().evaluate("REPLY",True,
            {"context_relevance_score":0.6,"novelty_score":0.8,"novelty_status":"ASSESSED"},
            True,True,"t+5m",observed_metrics={"likes":2,"replies":1})
        self.assertEqual(result.components["context_relevance_component"],0.6)
        self.assertGreater(result.components["observed_feedback_component"],0)
        self.assertTrue(result.calculation["engagement_counters_used"])
        self.assertFalse(result.calculation["parent_post_counters_used"])
        self.assertIn("novelty_component",result.calculation["analysis_only_components"])
        self.assertNotIn("novelty_component",result.calculation["used_components"])

    def test_11_api_failure_does_not_corrupt_action_history(self):
        plan=self.persist(make_plan())
        self.store.schedule_feedback(plan,ExecutionResult("SUCCEEDED","9001"),(300,))
        with self.store.connect() as db: db.execute("UPDATE action_feedback_snapshots SET scheduled_at='2000-01-01T00:00:00+00:00'")
        system=V03FeedbackSystem(FakeCounters(error=FakeHTTPError(404)),self.db,test_config())
        result=system.process_due()
        self.assertEqual(result["feedback_errors"],1)
        with self.history._connect() as db:
            self.assertEqual(db.execute("SELECT status FROM agent_actions WHERE action_id=?",(plan.action_id,)).fetchone()[0],"SUCCEEDED")
        with self.store.connect() as db:
            self.assertEqual(db.execute("SELECT feedback_status FROM action_feedback_snapshots").fetchone()[0],"UNAVAILABLE")

    def test_rate_limit_uses_server_retry_after_without_early_retry(self):
        plan=self.persist(make_plan())
        self.store.schedule_feedback(plan,ExecutionResult("SUCCEEDED","9001"),(300,))
        with self.store.connect() as db:
            db.execute("UPDATE action_feedback_snapshots SET scheduled_at='2000-01-01T00:00:00+00:00'")
        system=V03FeedbackSystem(FakeCounters(error=FakeHTTPError(429,600)),self.db,test_config())
        system.process_due()
        with self.store.connect() as db:
            row=db.execute("SELECT scheduled_at,attempt_count,feedback_status FROM action_feedback_snapshots").fetchone()
        retry=datetime.fromisoformat(row[0]); self.assertGreaterEqual((retry-datetime.now(timezone.utc)).total_seconds(),590)
        self.assertEqual(row[1],1); self.assertEqual(row[2],"PENDING")

    def test_wait_records_reason_and_next_observation_without_learning(self):
        plan=make_plan(action="WAIT",text=None)
        plan.reason="No eligible task; wait for next feed"
        self.history.save_result(plan,"WAIT")
        system=V03FeedbackSystem(FakeCounters(),self.db,test_config())
        system.record_plan(plan)
        system.record_result(plan,ExecutionResult("WAIT"))
        with self.store.connect() as db:
            db.execute("UPDATE action_feedback_snapshots SET scheduled_at='2000-01-01T00:00:00+00:00'")
        system.process_due()
        with self.store.connect() as db:
            feedback=db.execute("SELECT feedback_status,raw_metrics_json FROM action_feedback_snapshots").fetchone()
            reward=db.execute("SELECT reward_status,reward_total FROM action_rewards WHERE action_id=? ORDER BY computed_at DESC LIMIT 1",(plan.action_id,)).fetchone()
            update=db.execute("SELECT update_applied,update_reason FROM policy_updates WHERE action_id=?",(plan.action_id,)).fetchone()
        self.assertEqual(feedback[0],"COLLECTED"); self.assertIn("wait_reason",feedback[1])
        self.assertEqual(reward[0],"FINAL"); self.assertEqual(reward[1],1.0)
        self.assertEqual(update[0],0); self.assertEqual(update[1],"wait_baseline_not_learned")

    def test_12_reward_recalculation_is_idempotent_for_identical_inputs(self):
        evaluator=RewardEvaluator()
        result=evaluator.evaluate("CREATE_POST",True,{"novelty_score":0.8,"novelty_status":"ASSESSED",
            "source_alignment_score":0.4,"source_validation":{"source_count":1}},True,True,"t+6h")
        plan=self.persist(make_plan())
        first=self.store.save_reward(plan.action_id,"reward-v1-quality","t+6h",result.components,result.total,result.status,result.calculation)
        second=self.store.save_reward(plan.action_id,"reward-v1-quality","t+6h",result.components,result.total,result.status,result.calculation)
        self.assertEqual(first,second)

    def test_13_reward_versions_preserve_history(self):
        plan=self.persist(make_plan())
        result=RewardEvaluator().evaluate("LIKE",True,{},True,True,"execution")
        one=self.store.save_reward(plan.action_id,"reward-v1-quality","execution",result.components,result.total,"FINAL",result.calculation)
        two=self.store.save_reward(plan.action_id,"reward-v2-quality","execution",result.components,result.total,"FINAL",result.calculation)
        self.assertNotEqual(one,two)
        with self.store.connect() as db: self.assertEqual(db.execute("SELECT COUNT(*) FROM action_rewards").fetchone()[0],2)

    def test_14_learning_disabled_does_not_change_parameters(self):
        policy=HeuristicPolicy({"CREATE_POST":1.0})
        before=policy.get_parameters(); policy.observe_feedback("CREATE_POST",0.1); policy.update("CREATE_POST")
        self.assertEqual(before,policy.get_parameters())

    def test_15_parameter_updates_are_restorable(self):
        learner=ActionPriorLearningPolicy(("CREATE_POST","REPLY"),0.1)
        before=learner.get_parameters(); learner.observe_feedback("CREATE_POST",0.9)
        result=learner.update("CREATE_POST"); learner.save_state(self.store)
        restored=ActionPriorLearningPolicy(("CREATE_POST","REPLY"),0.1); restored.load_state(self.store)
        self.assertTrue(result["applied"]); self.assertNotEqual(before,restored.get_parameters())
        self.assertEqual(result["after"],restored.get_parameters())

    def test_learning_update_is_clipped_decayed_and_bounded(self):
        learner=ActionPriorLearningPolicy(("CREATE_POST",),learning_rate=1.0,
            max_update_step=0.01,baseline_decay=0.05,max_prior_deviation=0.1)
        learner.observe_feedback("CREATE_POST",1.0)
        first=learner.update("CREATE_POST")
        self.assertTrue(first["update_clipped"])
        self.assertAlmostEqual(abs(first["applied_delta"]),0.01,places=12)
        self.assertAlmostEqual(learner.prior_for("CREATE_POST"),1.01)
        for _ in range(60):
            learner.observe_feedback("CREATE_POST",1.0)
            learner.update("CREATE_POST")
        self.assertLessEqual(learner.prior_for("CREATE_POST"),1.1)
        before=learner.means["CREATE_POST"]
        learner.observe_feedback("CREATE_POST",0.5)
        learner.update("CREATE_POST")
        self.assertLess(learner.means["CREATE_POST"],before)

    def test_16_pending_feedback_survives_store_restart(self):
        plan=self.persist(make_plan())
        self.store.schedule_feedback(plan,ExecutionResult("SUCCEEDED","9001"),(300,))
        with self.store.connect() as db: db.execute("UPDATE action_feedback_snapshots SET scheduled_at='2000-01-01T00:00:00+00:00'")
        restarted=V03Store(self.db)
        self.assertEqual(len(restarted.due_feedback()),1)

    def test_17_analytics_exports_real_data_without_mock_claim(self):
        plan=self.persist(make_plan())
        self.store.record_decision(plan)
        from analytics.policy_report import run
        out=run(self.db,Path(self.temp.name)/"reports",
                configurations={"baseline":{"CREATE_POST":1.0},"lower_post":{"CREATE_POST":0.5}},
                source_provenance="test")
        self.assertTrue((out/"summary.json").exists())
        self.assertTrue((out/"actions.csv").exists())
        self.assertTrue((out/"reply_context_feedback.csv").exists())
        self.assertTrue((out/"prior_comparison.csv").exists())
        with (out/"prior_comparison.csv").open(encoding="utf-8-sig",newline="") as handle:
            prior_rows=list(__import__("csv").DictReader(handle))
        self.assertEqual(len(prior_rows),2)
        self.assertTrue(all(row["same_as_baseline"]=="True" for row in prior_rows))
        summary=json.loads((out/"summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["source_data_provenance"],"test")
        self.assertFalse(summary["mock_rows_generated_by_report"])
        self.assertTrue(summary["png_generated"])
        self.assertEqual((out/"policy_dashboard.png").read_bytes()[:8],b"\x89PNG\r\n\x1a\n")

    def test_18_dry_run_does_not_call_platform_write(self):
        class Account:
            def __init__(self): self.calls=0
            def post(self,body): self.calls+=1
        account=Account(); plan=make_plan()
        result=ActionExecutor(FakeCounters(),account,self.history).execute(plan,"dry-run")
        self.assertEqual(result.status,"DRY_RUN"); self.assertEqual(account.calls,0)

    def test_learning_cutoff_excludes_disabled_historical_rewards(self):
        plan = attach_valid_policy_features(make_plan())
        disabled = V03FeedbackSystem(FakeCounters({"like_count":3,"reply_count":1,
            "repost_count":0,"view_count":10}), self.db, test_config(enable_learning=False))
        disabled.record_plan(plan)
        self.history.save_result(plan, "SUCCEEDED", remote_id="9001")
        self.store.schedule_feedback(plan, ExecutionResult("SUCCEEDED", "9001"), (300,))
        with self.store.connect() as db:
            db.execute("UPDATE action_feedback_snapshots SET scheduled_at='2000-01-01T00:00:00+00:00'")
        disabled.process_due()
        self.assertEqual(disabled.bandit.count, 0)

        enabled = V03FeedbackSystem(FakeCounters({"like_count":3,"reply_count":1,
            "repost_count":0,"view_count":10}), self.db, test_config(windows=(300,), enable_learning=True))
        result = enabled.process_due()
        self.assertEqual(result["policy_updates"], 0)
        self.assertEqual(enabled.bandit.count, 0)
        self.assertIsNone(enabled.store.load_policy_state(FEATURE_VERSION))
        with self.store.connect() as db:
            updates = db.execute("SELECT update_applied,update_reason FROM policy_updates ORDER BY created_at").fetchall()
        self.assertEqual(sum(row[0] for row in updates), 0)
        self.assertIn("learning_disabled_fixed_baseline", [row[1] for row in updates])

        restarted = V03FeedbackSystem(FakeCounters({"like_count":3,"reply_count":1,
            "repost_count":0,"view_count":10}), self.db, test_config(windows=(300,), enable_learning=True))
        self.assertEqual(restarted.bandit.count, 0)
        restarted.process_due()
        self.assertEqual(restarted.bandit.count, 0)

    def test_non_text_engagement_reward_does_not_train_policy(self):
        plan = attach_valid_policy_features(make_plan(action="LIKE", text=None))
        system = V03FeedbackSystem(FakeCounters(), self.db, test_config(windows=(300,), enable_learning=True))
        system.record_plan(plan)
        self.history.save_result(plan, "SUCCEEDED", remote_id="9001")
        self.store.schedule_feedback(plan, ExecutionResult("SUCCEEDED", "9001"), (300,))
        with self.store.connect() as db:
            db.execute("UPDATE action_feedback_snapshots SET scheduled_at='2000-01-01T00:00:00+00:00'")
        system.process_due()
        self.assertEqual(system.bandit.count, 0)
        self.assertEqual(system.learning_policy.get_parameters()["counts"]["LIKE"], 0)
        with self.store.connect() as db:
            row = db.execute("SELECT update_applied,update_reason FROM policy_updates").fetchone()
        self.assertEqual(row[0], 0)
        self.assertEqual(row[1], "no_action_specific_observed_outcome")

    def test_interaction_frequency_counts_only_successful_recent_text_actions(self):
        from agent.runtime import AgentRuntime
        recent = [
            {"action_type":"CREATE_POST", "status":"SUCCEEDED"},
            {"action_type":"REPLY", "status":"SUCCEEDED"},
            {"action_type":"LIKE", "status":"SUCCEEDED"},
            {"action_type":"CREATE_POST", "status":"FAILED"},
        ]
        self.assertEqual(AgentRuntime._successful_text_actions_since_engagement(recent), 2)
        no_prior_engagement = [{"action_type":"REPLY", "status":"SUCCEEDED"}]
        self.assertIsNone(AgentRuntime._successful_text_actions_since_engagement(no_prior_engagement))
        dry_run_history = [
            {"action_type":"CREATE_POST", "status":"DRY_RUN"},
            {"action_type":"LIKE", "status":"SUCCEEDED"},
        ]
        self.assertEqual(AgentRuntime._successful_text_actions_since_engagement(dry_run_history), 0)
        self.assertEqual(AgentRuntime._successful_text_actions_since_engagement(
            dry_run_history, include_dry_runs=True), 1)

    def test_learning_disabled_writes_false_update_without_changing_parameters(self):
        plan=self.persist(make_plan())
        self.store.schedule_feedback(plan,ExecutionResult("SUCCEEDED","9001"),(300,))
        with self.store.connect() as db:
            db.execute("UPDATE action_feedback_snapshots SET scheduled_at='2000-01-01T00:00:00+00:00'")
        system=V03FeedbackSystem(FakeCounters({"like_count":3,"reply_count":1,
            "repost_count":0,"view_count":10}),self.db,test_config())
        system.process_due()
        with self.store.connect() as db:
            row=db.execute("SELECT update_applied,parameters_before_json,parameters_after_json FROM policy_updates").fetchone()
        self.assertEqual(row[0],0); self.assertEqual(row[1],row[2])

    def test_opt_in_quality_learning_persists_before_and_after(self):
        system=V03FeedbackSystem(FakeCounters({"like_count":3,"reply_count":1,
            "repost_count":0,"view_count":10}),self.db,test_config(enable_learning=True),
            {"CREATE_POST":1.0})
        plan=self.persist(make_plan())
        self.store.schedule_feedback(plan,ExecutionResult("SUCCEEDED","9001"),(300,))
        with self.store.connect() as db:
            db.execute("UPDATE action_feedback_snapshots SET scheduled_at='2000-01-01T00:00:00+00:00'")
        system.process_due()
        with self.store.connect() as db:
            row=db.execute("SELECT update_applied,parameters_before_json,parameters_after_json FROM policy_updates").fetchone()
        self.assertEqual(row[0],1); self.assertNotEqual(row[1],row[2])
        self.assertGreater(system.current_action_priors()["CREATE_POST"],1.0)

    def test_full_runtime_dry_run_uses_fake_observer_llm_and_database(self):
        from test_v02 import FakeObserver,FakeSocial,FakeLLM
        from agent.runtime import AgentRuntime
        from state.store import StateStore
        observer=FakeObserver(); social=FakeSocial(); llm=FakeLLM()
        history=ActionHistory(self.db); store=StateStore(self.db)
        raw_dir=Path(self.temp.name)/"raw"
        raw_dir.mkdir()
        def save_raw(snapshot):
            path=raw_dir/"snapshot.json"; path.write_text("{}",encoding="utf-8"); return path
        runtime=AgentRuntime(observer,store,history,social,llm_client=llm,mode="dry-run",
            max_actions_per_round=1,enable_topics=False,policy_mode="heuristic",
            runtime_config=test_config(),save_raw=save_raw)
        result=runtime.run_round()
        self.assertTrue(result["actions"])
        self.assertTrue(all(item["status"] in ("DRY_RUN","WAIT") for item in result["actions"]))
        with self.store.connect() as db:
            self.assertGreaterEqual(db.execute("SELECT COUNT(*) FROM policy_decisions").fetchone()[0],1)
        self.assertEqual(social.lock_calls,0)


if __name__=="__main__": unittest.main()
