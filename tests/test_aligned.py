import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from egip_aligned.actions import extract_action, execute_action
from egip_aligned.protocol import action_catalog
from egip_aligned.engine import run_episode, Config
from egip_aligned.finalizer import locked_gate, finalize, diagnosis
from egip_aligned.context_engineering import infer_model_strategy
from egip_aligned.native import command
from egip_aligned.run import ensure_manifest


class Scenario:
    scenario_dict = {'answers': [{'text': 'B', 'correct': False}, {'text': 'A', 'correct': True}]}
    def examiner_information(self): return 'Weakness for two days.'
    def patient_information(self): return {'history': 'PRIVATE_UNASKED_FACT'}
    def exam_information(self): return {'tests': {'Blood': 'RESULT_ONE', 'Imaging': 'UNREQUESTED_TWO'}}
    def diagnosis_information(self): raise AssertionError('Generation accessed evaluation gold')


def mapping(diagnosis='A', eid='e0002', insufficient=False):
    return json.dumps({'diagnosis': diagnosis, 'support': [{'event_id': eid, 'relation': 'observed pattern'}],
                      'conflicts': [], 'unknowns': [], 'insufficient': insufficient})


class Actions(unittest.TestCase):
    def setUp(self):
        self.exam = Scenario().exam_information()
        self.actions = [a.public for a in action_catalog(self.exam)]
        self.a, self.b = [a['action_id'] for a in self.actions]
    def test_dialogue_prefix_is_not_missing_record(self):
        a=extract_action("Let's do a test. REQUEST TEST: "+self.a,self.actions)
        self.assertEqual(a['kind'],'test')
        result=execute_action(a,self.exam)
        self.assertEqual(result['recorded_value'],'RESULT_ONE')
        self.assertNotIn('UNREQUESTED_TWO',result['text'])
    def test_case_insensitive_command_and_id(self):
        self.assertEqual(extract_action('request test: '+self.a.upper(),self.actions)['action_id'],self.a)
    def test_extra_lines_do_not_become_observations(self):
        a=extract_action('Some context.\nREQUEST TEST: '+self.a+'\nRESULTS: invented!',self.actions)
        self.assertNotIn('invented',execute_action(a,self.exam)['text'])
    def test_two_commands_rejected(self):
        self.assertEqual(extract_action('REQUEST TEST: '+self.a+'\nREQUEST TEST: '+self.b,self.actions)['kind'],'invalid')
    def test_two_ids_rejected(self):
        self.assertEqual(extract_action('REQUEST TEST: '+self.a+' and '+self.b,self.actions)['kind'],'invalid')
    def test_unknown_id_rejected(self):
        self.assertEqual(extract_action('REQUEST TEST: A_000000000000',self.actions)['kind'],'invalid')
    def test_value_cannot_be_requested(self):
        self.assertEqual(extract_action('REQUEST TEST: RESULT_ONE',self.actions)['kind'],'invalid')
    def test_full_public_label(self):
        self.assertEqual(extract_action('REQUEST TEST: '+self.actions[0]['name'],self.actions)['kind'],'test')
    def test_normal_question(self):
        self.assertEqual(extract_action('When did it begin?',self.actions)['kind'],'question')
    def test_id_before_command_not_accepted(self):
        self.assertEqual(extract_action(self.a+' is useful. REQUEST TEST: nonsense',self.actions)['kind'],'invalid')
    def test_bundle_value_is_returned_only_for_its_action(self):
        ex={'A':['one','two'],'B':'private'};a=action_catalog(ex)[0]
        self.assertNotIn('private',execute_action({'kind':'test','command':'REQUEST TEST: '+a.action_id},ex)['text'])


class Workflow(unittest.TestCase):
    def test_legacy_stage_order_and_closed_loop(self):
        stages=[];prompts={};actions=action_catalog(Scenario().exam_information());aid=actions[0].action_id
        def query(stage,prompt,system):
            stages.append(stage);prompts[stage]=prompt+system
            self.assertNotIn('UNREQUESTED_TWO',prompt+system)
            if not stage.startswith('patient'): self.assertNotIn('PRIVATE_UNASKED_FACT',prompt+system)
            if stage=='doctor_00':return "Let's check. REQUEST TEST: "+aid
            if stage=='doctor_01':
                self.assertIn('RESULT_ONE',prompt)
                # The remaining menu cannot advertise the already executed ID.
                self.assertNotIn(aid,system)
                return 'DIAGNOSIS READY: premature'
            if stage.startswith('patient'):return 'Observed onset yesterday.'
            if stage.startswith('council'):return 'Hypothesis A, supported by observed history.'
            if stage=='ledger':return 'positive_findings: RESULT_ONE\ncandidate_diagnoses: A'
            if stage=='consensus':return 'DIAGNOSIS READY: A'
            if stage=='evidence_gate':return mapping()
            raise AssertionError(stage)
        result=run_episode(Scenario(),'qwen2',query,Config(max_turns=3))
        self.assertEqual(stages,['doctor_00','doctor_01','patient_01','council_1','council_2','council_3','ledger','consensus','evidence_gate'])
        self.assertEqual(result['prediction'],'DIAGNOSIS READY: A')
        self.assertEqual(result['action_decisions'][1]['guard_reason'],'premature_finalization')
        self.assertFalse(result['clinical_entailment_verified'])
    def test_invalid_action_not_clinical_evidence(self):
        def query(stage,prompt,system):
            if stage=='doctor_00':return 'REQUEST TEST: A_000000000000'
            if stage=='doctor_01':
                self.assertIn('ACTION_REJECTED',prompt);self.assertIn('NOT a missing',prompt)
                return 'When did it begin?'
            if stage.startswith('patient'):return 'Yesterday.'
            if stage.startswith('council'):return 'A'
            if stage=='ledger':return 'Patient reported yesterday.'
            if stage=='consensus':return 'DIAGNOSIS READY: A'
            if stage=='evidence_gate':
                self.assertNotIn('ACTION_REJECTED',prompt)
                return mapping(eid='e0004')
        r=run_episode(Scenario(),'qwen2',query,Config(max_turns=3))
        self.assertFalse(any(e['speaker']=='Measurement' for e in r['events']))
        self.assertTrue(any(e['speaker']=='Interface' for e in r['events']))
        self.assertEqual(r['prediction'],'DIAGNOSIS READY: A')
    def test_repeat_converts_to_question_not_second_measurement(self):
        aid=action_catalog(Scenario().exam_information())[0].action_id
        def query(stage,prompt,system):
            if stage.startswith('doctor'):return 'REQUEST TEST: '+aid
            if stage.startswith('patient'):return 'Yesterday.'
            if stage.startswith('council'):return 'A'
            if stage=='ledger':return 'A'
            if stage=='consensus':return 'DIAGNOSIS READY: A'
            if stage=='evidence_gate':return mapping()
        r=run_episode(Scenario(),'qwen2',query,Config(max_turns=3))
        self.assertEqual(sum(e['speaker']=='Measurement' for e in r['events']),1)
        self.assertEqual(r['action_decisions'][1]['guard_reason'],'repeated_action')
    def test_large_audit_includes_actual_consensus(self):
        stages=[]
        def query(stage,prompt,system):
            stages.append(stage)
            if stage=='audit':self.assertIn('CURRENT_FINAL:\nDIAGNOSIS READY: A',prompt)
            if stage=='evidence_gate':return mapping(eid='initial_objective')
            return 'DIAGNOSIS READY: A'
        run_episode(Scenario(),'gpt4',query,Config(max_turns=1,strategy='large'))
        self.assertEqual(stages,['council_1','council_2','council_3','council_4','council_5','ledger','consensus','audit','evidence_gate'])
    def test_transport_error_does_not_become_evidence(self):
        with self.assertRaises(RuntimeError):
            run_episode(Scenario(),'qwen2',lambda *args:'Error: failed',Config(max_turns=2))
    def test_candidate_option_is_explicit(self):
        def query(stage,prompt,system):
            self.assertNotIn('"candidate_labels": ["A", "B"]',prompt)
            if stage=='evidence_gate':return mapping(eid='initial_objective')
            return 'DIAGNOSIS READY: A'
        r=run_episode(Scenario(),'qwen2',query,Config(max_turns=1,candidate_policy='none'))
        self.assertEqual(r['finalization']['evidence_interface']['candidate_labels'],[])


class Gate(unittest.TestCase):
    def setUp(self):
        self.e={'initial_objective':'weakness','candidate_labels':[],
                'observations':[{'event_id':'e0002','speaker':'Patient','text':'weakness'}]}
    def test_bold_consensus_preserves_disease_not_markup(self):
        self.assertEqual(diagnosis('**DIAGNOSIS READY: Disease A**'), 'Disease A')
        self.assertEqual(diagnosis('DIAGNOSIS READY: **Disease A**'), 'Disease A')
        answer,r=locked_gate(self.e,[], '**DIAGNOSIS READY: Disease A**',lambda *args:mapping('disease a'))
        self.assertEqual(answer,'DIAGNOSIS READY: Disease A')
        self.assertFalse(r['revision_applied'])
    def test_formatting_does_not_accept_different_disease(self):
        answer,r=locked_gate(self.e,[], '**DIAGNOSIS READY: Type 1 diabetes**',lambda *args:mapping('Type 2 diabetes'))
        self.assertEqual(answer,'DIAGNOSIS READY: ABSTAIN')
    def test_cannot_replace_consensus(self):
        calls=[]
        def q(*args):calls.append(args);return mapping('B')
        answer,r=locked_gate(self.e,[], 'DIAGNOSIS READY: A',q)
        self.assertEqual(answer,'DIAGNOSIS READY: ABSTAIN');self.assertEqual(len(calls),2)
        self.assertIn('diagnosis_must_equal_consensus_proposal',r['final_check']['errors'])
    def test_one_reference_repair(self):
        responses=iter([mapping(eid='bad'),mapping()]);calls=[]
        def q(*args):calls.append(args);return next(responses)
        answer,r=locked_gate(self.e,[],'DIAGNOSIS READY: A',q)
        self.assertEqual(answer,'DIAGNOSIS READY: A');self.assertEqual(len(calls),2)
    def test_model_insufficient_not_overridden(self):
        answer,r=locked_gate(self.e,[],'DIAGNOSIS READY: A',lambda *args:mapping(None,insufficient=True))
        self.assertEqual(answer,'DIAGNOSIS READY: ABSTAIN');self.assertFalse(r['revision_applied'])
    def test_unknown_cannot_support(self):
        self.e['observations'][0].update(speaker='Measurement',status='unavailable')
        answer,r=locked_gate(self.e,[],'DIAGNOSIS READY: A',lambda *args:mapping())
        self.assertEqual(answer,'DIAGNOSIS READY: ABSTAIN')
    def test_bad_medical_relation_not_claimed_verified(self):
        answer,r=locked_gate(self.e,[],'DIAGNOSIS READY: A',lambda *args:mapping())
        self.assertEqual(answer,'DIAGNOSIS READY: A');self.assertFalse(r['clinical_entailment_verified'])
    def test_no_hidden_extra_field(self):
        self.e['hidden_gold']='SECRET_GOLD';self.e['patient_raw']='SECRET_PATIENT'
        def q(stage,prompt,system):
            self.assertNotIn('SECRET_',prompt+system)
            return mapping() if stage=='evidence_gate' else 'DIAGNOSIS READY: A'
        finalize(self.e,infer_model_strategy('qwen2'),q)


class NativeAndConfiguration(unittest.TestCase):
    def test_native_command_only_passthrough(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve();(root/'medsim').mkdir();(root/'medsim/main.py').write_text('')
            args=['--total_inferences','20','--doctor_llm','local-vllm']
            cmd,cwd=command('MedAgentSim',root,'/native/python',args)
            self.assertEqual(cmd,['/native/python','-m','medsim.main']+args)
            self.assertEqual(cwd,root)
    def test_resume_refuses_changed_configuration(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);ensure_manifest(p,{'version':'A'},False)
            ensure_manifest(p,{'version':'A'},True)
            with self.assertRaises(ValueError):ensure_manifest(p,{'version':'B'},True)
    def test_no_silent_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'legacy.txt').write_text('old')
            with self.assertRaises(ValueError):ensure_manifest(p,{'version':'A'},False)


if __name__=='__main__':unittest.main()
