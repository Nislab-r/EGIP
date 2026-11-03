"""CLI integration against a synthetic external host, without GPU or network."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class CLI(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.host=self.root/'host'
        files={
            'medsim/__init__.py':'', 'medsim/core/__init__.py':'', 'medsim/main.py':'',
            'medsim/query_model.py': '''finalized=False
class BAgent:
 def __init__(self,model_name):self.model_name=model_name
 def query_model(self,prompt,system_prompt,image_requested=False):
  global finalized
  if 'diagnostic doctor' in system_prompt:
   finalized=True
   return 'DIAGNOSIS READY: Synthetic A'
  if 'role: patient' in system_prompt:return 'Synthetic symptom yesterday.'
  return 'When did it begin?'
''',
            'medsim/core/agent.py':'''def compare_results(prediction,gold,judge):
 return prediction=='DIAGNOSIS READY: '+gold
class MeasurementAgent:pass
''',
            'medsim/core/scenario.py':'''from medsim import query_model
def resolve_model_name(model):return model
class ScenarioMedQA:
 scenario_dict={'answers':[]}
 def __init__(self,data):pass
 def examiner_information(self):return 'Synthetic task.'
 def patient_information(self):return 'Private synthetic fact.'
 def exam_information(self):return {'Synthetic test':'Synthetic value'}
 def diagnosis_information(self):
  assert query_model.finalized, 'Gold accessed before prediction'
  return 'Synthetic A'
ScenarioMedQAExtended=ScenarioMIMICIVQA=ScenarioNEJM=ScenarioNEJMExtended=ScenarioMedQA
''',
            'datasets/_medqa.jsonl':'{}\n'}
        for name,value in files.items():
            p=self.host/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
        self.env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1')

    def invoke(self,module,*args):
        return subprocess.run([sys.executable,'-m',module,*map(str,args)],env=self.env,
                              capture_output=True,text=True,timeout=20)

    def test_run_resume_terminal_manifest_and_score(self):
        dest=self.root/'direct'
        args=['--host','MedAgentSim','--host-root',self.host,'--model','synthetic',
              '--dataset','MedQA','--max-turns','2','--output',dest]
        p=self.invoke('egip_aligned.run',*args);self.assertEqual(p.returncode,0,p.stderr)
        r=json.loads((dest/'case_0000/result.json').read_text())
        self.assertTrue(r['evaluation']['correct']);self.assertEqual(r['inference_calls'],3)
        self.assertEqual(r['finalization']['mode'],'direct_single_call')
        self.assertFalse(r['config']['evidence_gate'])
        p=self.invoke('egip_aligned.run',*args,'--resume');self.assertEqual(p.returncode,0,p.stderr)
        p=self.invoke('egip_aligned.run',*args,'--resume','--terminal','vote')
        self.assertNotEqual(p.returncode,0)
        source=dest/'case_0000/result.json'
        p=self.invoke('egip_aligned.replay','--source',source,'--host','MedAgentSim',
                      '--host-root',self.host,'--model','synthetic','--terminal','vote',
                      '--first-direct',source,'--output',self.root/'vote')
        self.assertEqual(p.returncode,0,p.stderr)
        v=json.loads((self.root/'vote/result.json').read_text())
        self.assertEqual(v['inference_calls'],4);self.assertIsNone(v['evaluation'])
        self.assertEqual(v['finalization']['evidence_interface'],r['finalization']['evidence_interface'])

        p=self.invoke('egip_aligned.replay','--source',source,'--host','MedAgentSim',
                      '--host-root',self.host,'--model','synthetic','--terminal','direct',
                      '--arm','no_observations','--score','--dataset','MedQA','--case-id','0',
                      '--output',self.root/'removed')
        self.assertEqual(p.returncode,0,p.stderr)
        scored=json.loads((self.root/'removed/result.json').read_text())
        self.assertTrue(scored['evaluation']['correct'])
        self.assertEqual(scored['finalization']['evidence_interface']['observations'],[])

    def test_incompatible_options_fail_before_output(self):
        dest=self.root/'bad'
        p=self.invoke('egip_aligned.run','--host','MedAgentSim','--host-root',self.host,
                      '--model','synthetic','--dataset','MedQA','--terminal','vote',
                      '--acquisition-ablation','no_action_menu','--output',dest)
        self.assertNotEqual(p.returncode,0);self.assertFalse(dest.exists())
