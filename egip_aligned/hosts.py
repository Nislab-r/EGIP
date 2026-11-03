"""Connect EGIP to MedAgentSim scenarios, model calls and scoring."""
import importlib
from pathlib import Path
import sys

BUNDLED_ROOT = Path(__file__).parent / '_medagentsim'
CLASSES = {'MedQA': 'ScenarioMedQA', 'MedQA_ext': 'ScenarioMedQAExtended',
           'MIMIC-IV': 'ScenarioMIMICIVQA', 'NEJM': 'ScenarioNEJM', 'NEJM_ext': 'ScenarioNEJMExtended'}
MAS_FILES = {'MedQA': '_medqa.jsonl', 'MedQA_ext': '_medqa_extended.jsonl',
             'MIMIC-IV': 'datasets_mimiciv.jsonl', 'NEJM': '_nejm.jsonl', 'NEJM_ext': 'datasets_nejm_extended.jsonl'}

class Host:
    def __init__(self, name, root, model, moderator=None):
        if name != 'MedAgentSim':
            raise ValueError('This release supports MedAgentSim only')
        self.name, self.root = name, Path(root).resolve()
        sys.path.insert(0, str(self.root))
        self.transport = importlib.import_module('medsim.query_model')
        self.agents = importlib.import_module('medsim.core.agent')
        self.scenarios = importlib.import_module('medsim.core.scenario')
        for module in (self.transport, self.agents, self.scenarios):
            if not Path(module.__file__).resolve().is_relative_to(self.root):
                raise RuntimeError('MedAgentSim already loaded from another directory; start a fresh process')
        self.model = self.scenarios.resolve_model_name(model)
        self.moderator = self.scenarios.resolve_model_name(moderator or model)
        self.generator = self.transport.BAgent(model_name=self.model)
        self.judge = self.transport.BAgent(model_name=self.moderator)

    def data_path(self, dataset):
        return self.root / 'datasets' / MAS_FILES[dataset]

    def scenario(self, dataset, data):
        return getattr(self.scenarios, CLASSES[dataset])(data)

    def query(self, prompt, system):
        return self.generator.query_model(prompt, system, image_requested=False)

    def score(self, prediction, gold):
        return bool(self.agents.compare_results(prediction, gold, self.judge))
