import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from statelm.core import Inventory,FSM,STRATEGIES,select_recovery
from statelm.slm import preference_loss,recovery_training_loss,validation_metrics,select_checkpoint,HFSLM
from statelm.playground import Session
from test_core import ScriptedSLM

ROOT=Path(__file__).resolve().parents[1]

class TrainingRevisionTests(unittest.TestCase):
    def test_one_confident_selected_strategy_is_enough(self):
        logits=torch.tensor([12.,-4.,-4.,-4.,-4.,-4.,-4.,-4.],requires_grad=True)
        loss=preference_loss(logits,[0,2,5]);loss.backward()
        self.assertLess(float(loss),.0001)
        self.assertLess(float(logits.grad.abs().max()),.0001)
    def test_wrong_strategy_gets_a_gradient_toward_preference_set(self):
        logits=torch.tensor([9.,0.,0.,0.,0.,0.,0.,0.],requires_grad=True)
        preference_loss(logits,[1,4]).backward()
        self.assertGreater(float(logits.grad[0]),0)
        self.assertLess(float(logits.grad[1]),0);self.assertLess(float(logits.grad[4]),0)
    def test_anchor_preserves_base_without_training_reference(self):
        reference=torch.tensor([4.,2.,0.,0.,0.,0.,0.,0.],requires_grad=True)
        original=reference.detach().clone().requires_grad_()
        drifted=torch.zeros(8,requires_grad=True)
        all_choices=list(range(8))
        self.assertAlmostEqual(float(recovery_training_loss(original,all_choices,reference)),0.,places=6)
        loss=recovery_training_loss(drifted,all_choices,reference);loss.backward()
        self.assertGreater(float(loss),0);self.assertIsNone(reference.grad)
        self.assertLess(float(drifted.grad[0]),0)
    def test_base_wins_ties_and_validation_regressions(self):
        base={'n_cases':20,'n_correct':14}
        self.assertEqual(select_checkpoint(base,{'n_cases':20,'n_correct':9}),'base')
        self.assertEqual(select_checkpoint(base,{'n_cases':20,'n_correct':14}),'base')
        self.assertEqual(select_checkpoint(base,{'n_cases':20,'n_correct':15}),'adapter')
        with self.assertRaises(ValueError):select_checkpoint(base,{'n_cases':19,'n_correct':15})
    def test_validation_uses_deployed_modality_rule(self):
        logits=torch.tensor([0.,8.,0.,0.,0.,0.,0.,10.])
        result=validation_metrics([{'logits':logits,'selected_indices':[1],'voice_and_display_available':False}])
        self.assertEqual(result['n_correct'],1);self.assertEqual(result['raw_agreement'],0)
        self.assertEqual(result['strategy_counts'],{'self_correction':1})
    def test_validation_selection_disables_only_recovery_adapter(self):
        from contextlib import contextmanager
        from unittest.mock import patch
        class Model:
            disabled=False
            def eval(self):pass
            @contextmanager
            def disable_adapter(self):
                self.disabled=True
                try:yield
                finally:self.disabled=False
        model=Model();slm=HFSLM(None,model,{'max_length':2048},True)
        def logits(m,*_):return torch.tensor([9.,0.,0.,0.,0.,0.,0.,0.]) if m.disabled else torch.tensor([0.,9.,0.,0.,0.,0.,0.,0.])
        with patch('statelm.slm.encode_choices',return_value=([1],[1,2,3,4,5,6,7,8])),patch('statelm.slm.choice_logits',side_effect=logits):
            h=[{'role':'participant','text':'scissors'}]
            slm.selected_recovery_model='base';slm.use_validation_selection=True
            self.assertEqual(slm.recover(h,'lab',[])['strategy'],'clarifying_prompt')
            slm.use_validation_selection=False
            self.assertEqual(slm.recover(h,'lab',[])['strategy'],'self_correction')
        self.assertFalse(model.disabled)

class PlaygroundTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.slm=ScriptedSLM();self.slm.selected_recovery_model='adapter'
        self.inv=Inventory(ROOT/'data/inventory.json')
        cases=json.loads((ROOT/'data/original_cases.json').read_text())['cases']
        self.case=next(c for c in cases if c['task']=='recovery')
        self.session=Session(self.inv,FSM(),self.slm,{'threshold':.98,'margin':.8},[self.case],
                            Path(self.tmp.name)/'interactive.jsonl',{'fold':0})
    def test_each_comparison_runs_models_and_rating_does_not_retrain(self):
        result=self.session.handle({'op':'message','cart':'lab','text':'scisor'})
        self.assertIn('slm',result['outputs']);self.assertIsNone(result['targets'])
        before=list(self.slm.calls)
        self.session.handle({'op':'rate','condition':'slm','rating':'appropriate'})
        self.assertEqual(self.slm.calls,before)
        self.session.handle({'op':'reset'})
        self.session.handle({'op':'message','cart':'lab','text':'scisor'})
        self.assertGreater(len(self.slm.calls),len(before))
    def test_continue_does_not_make_hybrid_sticky(self):
        self.session.handle({'op':'message','cart':'lab','text':'scisor'})
        self.session.handle({'op':'continue','condition':'slm'})
        result=self.session.handle({'op':'message','cart':'lab','text':'Which drawer?'})
        self.assertEqual(result['outputs']['hybrid']['output']['route'],'fsm')
        self.assertEqual(result['outputs']['hybrid']['output']['drawer'],5)
    def test_own_interaction_requires_reply_choice(self):
        self.session.handle({'op':'message','cart':'lab','text':'scissors'})
        with self.assertRaises(ValueError):self.session.handle({'op':'message','cart':'lab','text':'blue tape'})
        self.session.handle({'op':'reset'})
        r=self.session.handle({'op':'message','cart':'lab','text':'blue tape'})
        self.assertEqual(r['outputs']['hybrid']['output']['drawer'],1)
    def test_replay_is_heldout_only_and_does_not_change_case(self):
        before=copy.deepcopy(self.case)
        with self.assertRaises(KeyError):self.session.handle({'op':'replay','case_id':'not-a-heldout-case'})
        out=self.session.handle({'op':'replay','case_id':self.case['case_id']})
        self.assertEqual(self.case,before);self.assertEqual(out['targets'],self.case['targets'])
        self.assertIsNotNone(out['outputs']['hybrid']['metrics'])
        entries=[json.loads(s) for s in self.session.log_path.read_text().splitlines()]
        self.assertTrue(all(e['suite']=='interactive_only' for e in entries))

if __name__=='__main__':unittest.main()
