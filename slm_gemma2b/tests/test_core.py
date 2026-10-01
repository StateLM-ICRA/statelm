import copy
from pathlib import Path
import unittest
from statelm.core import Agent,Inventory,FSM,STRATEGIES,drawer_mentions,location,failure_evidence
from statelm.evaluation import score,summarize,run_cases
from statelm.data import read,require_original,make_test_n,fold_plan,split_cases
ROOT=Path(__file__).resolve().parents[1]

class ScriptedSLM:
    """Unit-test double only. Not available from any experiment CLI."""
    def __init__(self,item='scissors',p=.999,strategy='self_correction',action='locate'):
        self.item=item;self.p=p;self.strategy=strategy;self.action=action;self.calls=[]
    def resolve(self,history,cart,inventory):
        self.calls.append('resolve');return {'item_id':self.item,'action':self.action,'probability':self.p,'margin':self.p-.001,'forward_calls':1}
    def recover(self,*args):
        self.calls.append('recover');return {'strategy':self.strategy,'scores':{s:1. if s==self.strategy else .01 for s in STRATEGIES},'forward_calls':1}

class CoreTests(unittest.TestCase):
    def setUp(self):self.inv=Inventory(ROOT/'data/inventory.json')
    def inp(self,text,cart='lab'):return {'cart':cart,'history':[{'role':'participant','text':text}]}
    def test_all_inventory_entries_are_exact_and_cart_specific(self):
        for x in self.inv.items:
            a=Agent(self.inv,condition='fsm');o=a.reply(self.inp('Where is '+x['name']+'?',x['setting']))
            self.assertEqual((o['item_id'],o['drawer']),(x['item_id'],x['drawer']))
        self.assertEqual(self.inv.get('lab','scissors')['drawer'],5)
        self.assertEqual(self.inv.get('hospital','scissors')['drawer'],4)
    def test_known_item_never_calls_slm(self):
        slm=ScriptedSLM();a=Agent(self.inv,slm=slm)
        self.assertEqual(a.reply(self.inp('Where is blue tape?'))['drawer'],1);self.assertEqual(slm.calls,[])
    def test_unknown_item_calls_slm_then_uses_table(self):
        slm=ScriptedSLM();a=Agent(self.inv,slm=slm)
        o=a.reply(self.inp('Where is scisor?'));self.assertEqual(o['drawer'],5);self.assertEqual(slm.calls,['resolve'])
    def test_low_probability_clarifies(self):
        a=Agent(self.inv,slm=ScriptedSLM(p=.4));o=a.reply(self.inp('a round white object'))
        self.assertEqual(o['action'],'clarify');self.assertIsNone(o['drawer'])
    def test_invalid_model_item_cannot_invent_drawer(self):
        o=Agent(self.inv,slm=ScriptedSLM(item='imaginary')).reply(self.inp('gizmo'))
        self.assertEqual(o['action'],'clarify');self.assertIsNone(o['drawer'])
    def test_slm_cannot_override_an_explicit_other_item(self):
        o=Agent(self.inv,slm=ScriptedSLM(item='scissors'),condition='slm').reply(self.inp('Where is blue tape?'))
        self.assertEqual(o['action'],'clarify');self.assertIsNone(o['drawer'])
    def test_exact_model_proposal_is_independently_confirmed(self):
        o=Agent(self.inv,slm=ScriptedSLM(item='scissors'),condition='slm',resolve_threshold=1.01).reply(self.inp('scissors'))
        self.assertEqual(o['drawer'],5)
    def test_multi_item_or_negated_request_not_located(self):
        for text in ('scissors or book','not scissors','scissors and book'):
            o=Agent(self.inv,slm=ScriptedSLM()).reply(self.inp(text));self.assertEqual(o['action'],'clarify')
    def test_recovery_uses_model_and_does_not_stick(self):
        slm=ScriptedSLM();a=Agent(self.inv,slm=slm)
        inp=self.inp('Where are the scissors?');inp['history'].append({'role':'robot','text':'Open the drawer'})
        o=a.reply(inp);self.assertEqual(o['strategy'],'self_correction');self.assertIn('recover',slm.calls)
        o=a.reply(self.inp('Where is blue tape?'));self.assertEqual(o['route'],'fsm');self.assertEqual(o['drawer'],1)
    def test_failure_flag_is_not_required_for_observed_failure(self):
        h=[{'role':'participant','text':'blue tape'},{'role':'robot','text':'I think it is in drawer two'}]
        self.assertTrue(failure_evidence(h,self.inv,'lab'))
    def test_followup_preserves_item_new_unknown_clears_it(self):
        a=Agent(self.inv,condition='fsm');a.reply(self.inp('scissors'))
        self.assertEqual(a.reply(self.inp('Which drawer?'))['drawer'],5)
        self.assertEqual(a.reply(self.inp('unknown widget'))['action'],'clarify')
        self.assertIsNone(a.reply(self.inp('Which drawer?'))['drawer'])
    def test_plural_and_ordinal_scoring(self):
        self.assertEqual(location(self.inv.get('lab','wires')),'The wires are in drawer 3.')
        self.assertEqual(drawer_mentions('in the second drawer, not drawer 1'),[2,1])
    def test_denial_and_question_never_score_as_location(self):
        c={'task':'retrieval','targets':{'item_id':'scissors','drawer':5,'acceptable_actions':['locate']}}
        for text in ('The scissors are not in drawer five.','Are the scissors in drawer five?','drawer on'):
            o={'action':'locate','item_id':'scissors','drawer':5,'text':text}
            self.assertFalse(score(c,o)['joint_item_drawer_correct'])
    def test_recovery_scorer_accepts_any_selected_preference(self):
        c={'task':'recovery','targets':{'preferences':['self_correction','confidence_check'],'acceptable_actions':['recover']}}
        for s in c['targets']['preferences']:
            self.assertTrue(score(c,{'action':'recover','strategy':s,'raw_strategy':s})['preference_agreement'])
    def test_multimodal_is_not_claimed_without_capability(self):
        a=Agent(self.inv,slm=ScriptedSLM(strategy='multimodal_redundancy'))
        i=self.inp('scissors');i['recovery_requested']=True;o=a.reply(i)
        self.assertEqual(o['raw_strategy'],'multimodal_redundancy');self.assertNotEqual(o['strategy'],'multimodal_redundancy')
    def test_unknown_report_is_explicit_and_cannot_overwrite(self):
        with self.assertRaises(ValueError):self.inv.add_report('lab','new widget',2,'I found it in drawer two')
        x=self.inv.add_report('lab','new widget',2,'I found it in drawer two',enabled=True)
        self.assertEqual(self.inv.exact('new widget','lab')[0]['drawer'],2)
        with self.assertRaises(ValueError):self.inv.add_report('lab','scissors',1,'not there',enabled=True)
        with self.assertRaises(ValueError):self.inv.add_report('lab','another widget',6,'drawer six',enabled=True)
    def test_absent_model_is_an_error_not_a_fake_result(self):
        with self.assertRaises(RuntimeError):Agent(self.inv).reply(self.inp('unrecognized request'))

class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=read(ROOT/'data/original_cases.json');cls.cases=cls.data['cases']
    def test_source_suite_unchanged_and_no_targets_in_input(self):
        require_original(self.cases)
        for c in self.cases:
            self.assertFalse(c['synthetic']);self.assertNotIn('preferences',c['input']);self.assertNotIn('item_id',c['input'])
            self.assertNotIn('failure_types',c['input']);self.assertNotIn('failure_end',c['input'])
    def test_participant_and_trial_splits_do_not_overlap(self):
        for f in fold_plan(self.cases):
            parts=split_cases(self.cases,f)
            ids=[{c['participant_id'] for c in parts[k]} for k in ('train','validation','test')]
            self.assertFalse(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
    def test_noise_does_not_modify_original_and_is_test_only(self):
        before=copy.deepcopy(self.cases);parts=split_cases(self.cases,fold_plan(self.cases)[0]);n=make_test_n(parts['test'])
        self.assertEqual(self.cases,before);self.assertTrue(n)
        with self.assertRaises(ValueError):require_original(n)
        with self.assertRaises(ValueError):FSM().fit(n)
        with self.assertRaises(ValueError):run_cases(n,Inventory(ROOT/'data/inventory.json'),FSM(),'fsm')
        parent={c['case_id'] for c in parts['test']}
        self.assertTrue(all(x['provenance']['parent_case_id'] in parent for x in n))
    def test_cannot_pool_original_and_noise_scores(self):
        with self.assertRaises(ValueError):summarize([{'suite':'original'},{'suite':'test_n'}],'original')
    def test_one_decision_per_task_per_physical_interaction(self):
        keys=[(c['trial_id'],c['task']) for c in self.cases];self.assertEqual(len(keys),len(set(keys)))
    def test_failure_prefix_excludes_later_correct_reply(self):
        c=next(c for c in self.cases if 'sample_037' in c['provenance']['sample_aliases'] and c['task']=='recovery')
        self.assertIn('not understand',c['input']['history'][-1]['text'])
        self.assertFalse(any('drawer five' in t['text'].lower() for t in c['input']['history']))

if __name__=='__main__':unittest.main()
