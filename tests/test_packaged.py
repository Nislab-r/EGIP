"""Integration tests for the self-contained MedAgentSim/MedQA example."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from egip_aligned.hosts import BUNDLED_ROOT
from http_fixture import mock_server
from egip_aligned.summarize import aggregate

ROOT = Path(__file__).resolve().parents[1]

class Packaged(unittest.TestCase):
    def invoke(self, module, url, *args):
        env=dict(os.environ,PYTHONPATH=str(ROOT),EGIP_API_BASE=url,
                 EGIP_DISABLE_THINKING='1',EGIP_TEMPERATURE='0.2',EGIP_MAX_TOKENS='200')
        return subprocess.run([sys.executable,'-m',module,*map(str,args)],
                              env=env,capture_output=True,text=True,timeout=30)

    def test_reference_data_has_107_cases_and_required_fields(self):
        records=[json.loads(line) for line in (BUNDLED_ROOT/'datasets/_medqa.jsonl').read_text().splitlines()]
        self.assertEqual(len(records),107)
        for record in records:
            self.assertTrue({'Test_Results','Correct_Diagnosis','Patient_Actor',
                             'Objective_for_Doctor','Physical_Examination_Findings'} <= record['OSCE_Examination'].keys())

    def test_three_repetitions_real_runner_and_http_payload(self):
        with tempfile.TemporaryDirectory() as directory, mock_server() as (url,calls):
            p=self.invoke('egip_aligned.reproduce',url,'--model','mock','--case-ids','0',
                          '--output',directory)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            root=Path(directory);summary=json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['repetitions'],3)
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['sample_sd_pct'],0)
            for i in range(1,4):
                result=json.loads((root/f'repeat_{i}/case_0000/result.json').read_text())
                self.assertEqual(len(result['action_decisions']),19)
                self.assertEqual(result['config']['max_turns'],20)
                self.assertNotIn('source_sha256',json.dumps(result))
                self.assertNotIn('input_sha256',json.dumps(result))
                self.assertEqual(result['finalization']['evidence_interface']['candidate_labels'],[])
            for request in calls:
                self.assertEqual(request['model'],'mock')
                self.assertEqual(request['max_tokens'],200)
                self.assertEqual(request['temperature'],0.2)
                self.assertEqual(request['chat_template_kwargs'],{'enable_thinking':False})
                self.assertNotIn('seed',request)
            scores=[r for r in calls if 'medical moderator' in r['messages'][0]['content']]
            self.assertEqual(len(scores),3)
            self.assertTrue(all('Here is the correct diagnosis:' in r['messages'][1]['content'] for r in scores))
            self.assertFalse(any('Here is the correct diagnosis:' in r['messages'][1]['content'] for r in calls if r not in scores))

    def test_bundled_native_measurement_ablation_uses_original_prompt(self):
        with tempfile.TemporaryDirectory() as directory, mock_server() as (url,calls):
            p=self.invoke('egip_aligned.run',url,'--model','mock','--case-ids','0',
                          '--max-turns','2','--acquisition-ablation','native_measurement','--output',directory)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            result=json.loads((Path(directory)/'case_0000/result.json').read_text())
            observation=result['finalization']['evidence_interface']['observations'][0]
            self.assertEqual(observation['status'],'native_generated')
            prompts=[r['messages'][0]['content'] for r in calls]
            self.assertTrue(any('NORMAL READINGS' in s for s in prompts))

    def test_unknown_model_fails_without_generation(self):
        with tempfile.TemporaryDirectory() as directory, mock_server() as (url,calls):
            p=self.invoke('egip_aligned.run',url,'--model','not-served',
                          '--case-ids','0','--output',directory)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('not served',p.stderr)
            self.assertEqual(calls,[])

    def test_aggregate_sample_sd_and_incompatible_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            roots=[]
            for i,corrects in enumerate(([True,False],[True,True])):
                root=Path(directory)/str(i);root.mkdir();roots.append(root)
                (root/'settings.json').write_text(json.dumps({'case_ids':[0,1],'model':'a'}))
                for j,correct in enumerate(corrects):
                    case=root/f'case_{j:04d}';case.mkdir()
                    (case/'result.json').write_text(json.dumps({'evaluation':{'correct':correct}}))
            summary=aggregate(roots)
            self.assertEqual(summary['mean_accuracy_pct'],75)
            self.assertAlmostEqual(summary['sample_sd_pct'],35.3553390593)
            (roots[1]/'settings.json').write_text(json.dumps({'case_ids':[0,1],'model':'b'}))
            with self.assertRaises(ValueError):aggregate(roots)

if __name__=='__main__':
    unittest.main()
