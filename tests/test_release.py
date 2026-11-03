import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from egip_aligned import engine
from egip_aligned.terminals import direct, vote, PREFIX, SYSTEM
from egip_aligned.ablations import acquisition_intervention, remove_terminal_input
from egip_aligned.summarize import summarize
from test_aligned import Scenario


class Release(unittest.TestCase):
    def setUp(self):
        self.ev=dict(initial_objective='Synthetic task',candidate_labels=['A','B'],observations=[])

    def test_direct_allowlist_and_exact_prompt(self):
        ev=dict(self.ev,hidden='must not pass')
        def q(stage,prompt,system):
            self.assertEqual(stage,'direct_0');self.assertEqual(system,SYSTEM)
            self.assertEqual(prompt,PREFIX+json.dumps(self.ev,ensure_ascii=False))
            return 'DIAGNOSIS READY: A'
        _,record=direct(ev,None,q)
        self.assertNotIn('hidden',record['evidence_interface'])

    def test_vote_reuses_first_and_first_label_wins_tie(self):
        answers=iter(['DIAGNOSIS READY: B','DIAGNOSIS READY: A','DIAGNOSIS READY: B','unparseable'])
        calls=[]
        def q(stage,*args):calls.append(stage);return next(answers)
        answer,rec=vote(self.ev,None,q,first_answer='DIAGNOSIS READY: A')
        self.assertEqual(answer,'DIAGNOSIS READY: a');self.assertEqual(len(calls),4)
        self.assertEqual(len(rec['vote_members']),5)

    def test_vote_abstains_without_valid_votes(self):
        answer,_=vote(self.ev,None,lambda *args:'DIAGNOSIS READY: ABSTAIN')
        self.assertEqual(answer,'DIAGNOSIS READY: ABSTAIN')

    def test_terminal_removal_does_not_mutate_original(self):
        ev=copy.deepcopy(self.ev);ev['observations']=[{'text':'synthetic'}]
        out=remove_terminal_input(ev,'no_observations')
        self.assertEqual(out['observations'],[]);self.assertEqual(len(ev['observations']),1)
        self.assertEqual(out['candidate_labels'],ev['candidate_labels'])
        with self.assertRaises(ValueError):remove_terminal_input(dict(ev,candidate_labels=[]),'no_candidate_labels')

    def test_direct_never_accesses_gold(self):
        stages=[]
        def q(stage,*args):
            stages.append(stage)
            return 'DIAGNOSIS READY: A' if stage=='direct_0' else 'When did it begin?' if stage.startswith('doctor_') else 'Yesterday.'
        result=engine.run_episode(Scenario(),'model',q,engine.Config(max_turns=2,evidence_gate=False),terminal=direct)
        self.assertEqual(stages,['doctor_00','patient_00','direct_0'])
        self.assertEqual(result['prediction'],'DIAGNOSIS READY: A')

    def test_menu_ablation_removes_display_and_restores_engine(self):
        original=engine.plan;seen=[]
        def q(stage,prompt,system):
            seen.append(system)
            if stage=='doctor_00':return 'REQUEST TEST: Blood'
            if stage=='direct_0':return 'DIAGNOSIS READY: A'
            raise AssertionError(stage)
        with acquisition_intervention('no_action_menu',None,None,q) as query:
            r=engine.run_episode(Scenario(),'model',query,engine.Config(max_turns=2,evidence_gate=False),terminal=direct)
        self.assertIs(engine.plan,original)
        self.assertNotIn('AVAILABLE_TEST_MENU_NAMES_ONLY',seen[0])
        self.assertEqual(r['events'][1]['recorded_value'],'RESULT_ONE')

    def test_ablation_restores_after_error(self):
        original=engine.plan
        with self.assertRaises(RuntimeError):
            with acquisition_intervention('no_action_menu',None,None,lambda *x:None):raise RuntimeError('synthetic')
        self.assertIs(engine.plan,original)

    def test_native_measurement_adapter_calls_host(self):
        class Measurement:
            def update_scenario(self,scenario):self.scenario=scenario
            def inference_measurement(self,request):return self.pipe.query_model(request,'synthetic measurement prompt')
        host=SimpleNamespace(model='synthetic',agents=SimpleNamespace(MeasurementAgent=Measurement))
        seen=[]
        def q(*args):seen.append(args);return 'synthetic native finding'
        original=engine.execute_action
        with acquisition_intervention('native_measurement',host,Scenario(),q):
            record=engine.execute_action({'name':'Blood','action_id':'A_test'},None)
        self.assertEqual(record['status'],'native_generated');self.assertEqual(seen[0][0],'native_measurement')
        self.assertIs(engine.execute_action,original)

    def test_summary_retains_missing_denominator(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'settings.json').write_text(json.dumps({'case_ids':[0,1,2]}))
            case=root/'case_0000';case.mkdir();(case/'result.json').write_text(json.dumps({'evaluation':{'correct':True}}))
            other=root/'case_0001';other.mkdir();(other/'result.json').write_text(json.dumps({'evaluation':{'correct':None}}))
            r=summarize(root);self.assertAlmostEqual(r['accuracy_pct'],100/3)
            self.assertEqual(r['unscored_present'],1);self.assertEqual(r['missing_or_incomplete'],1)

    def test_acquisition_matches_full_with_different_terminal(self):
        calls=[]
        def q(stage,prompt,system):
            calls.append((stage,prompt,system))
            if stage.startswith('doctor_'):return 'When did it begin?'
            if stage.startswith('patient_'):return 'Yesterday.'
            return 'DIAGNOSIS READY: A'
        cfg=engine.Config(max_turns=3,evidence_gate=False)
        a=engine.run_episode(Scenario(),'model',q,cfg,terminal=direct);one=calls[:4];calls.clear()
        b=engine.run_episode(Scenario(),'model',q,cfg)
        self.assertEqual(one,calls[:4]);self.assertEqual(a['finalization']['evidence_interface'],b['finalization']['evidence_interface'])
