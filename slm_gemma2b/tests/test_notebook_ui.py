"""Exercise actual widget callbacks with a test-only backend; no benchmark scores."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from statelm.core import Inventory,FSM
from statelm.playground import Session
from statelm.notebook_ui import Playground,ModelProcess
from test_core import ScriptedSLM

ROOT=Path(__file__).resolve().parents[1]

class WidgetTests(unittest.TestCase):
    def test_load_compare_review_continue_and_reset_buttons(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run=root/'old_run';(run/'seed_42/fold_0/adapter').mkdir(parents=True)
            (run/'config.json').write_text('{}');(run/'seed_42/fold_0/adapter/training_record.json').write_text('{}')
            slm=ScriptedSLM();slm.selected_recovery_model='adapter'
            case=next(c for c in json.loads((ROOT/'data/original_cases.json').read_text())['cases'] if c['task']=='recovery')
            session=Session(Inventory(ROOT/'data/inventory.json'),FSM(),slm,{'threshold':.98,'margin':.8},[case],root/'interactive.jsonl',
                {'fold':0,'seed':42,'model_load_seconds':1.,'warmup_seconds':1.})
            class Backend:
                def __init__(self,*_):self.metadata=session.handle({'op':'metadata'})
                def request(self,obj):return session.handle(obj)
                def stop(self):pass
            with patch('statelm.notebook_ui.ModelProcess',Backend),patch('IPython.display.display'):
                ui=Playground(sys.executable,ROOT,[('Original',run)],root/'logs')
                ui.load_button.click();self.assertIsNotNone(ui.backend)
                ui.message.value='scisor';ui.compare_button.click()
                self.assertEqual(len(ui.results.children[0].children),4)
                trained_card=ui.results.children[0].children[2]
                trained_card.children[1].value='appropriate';trained_card.children[3].click()
                self.assertIn('saved separately',ui.notice.value)
                trained_card.children[4].click();self.assertIn('follow-up',ui.notice.value)
                ui.message.value='Which drawer?';ui.compare_button.click()
                self.assertEqual(session.pending['outputs']['hybrid']['output']['route'],'fsm')
                ui.new_button.click();self.assertEqual(session.history,[])
                ui.replay_button.click();self.assertIn('Recorded preferences',ui.notice.value)
                ui.unload_button.click();self.assertIsNone(ui.backend)
    def test_worker_load_failure_is_an_error_not_fake_output(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError):ModelProcess(sys.executable,ROOT,Path(td)/'missing',0,42,Path(td)/'logs')

if __name__=='__main__':unittest.main()
