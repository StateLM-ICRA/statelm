"""Resident real-model tester. Its logs never modify benchmark or training data."""
import argparse
import copy
from contextlib import redirect_stdout
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import uuid
from .core import Agent,FSM,Inventory,digest
from .data import read,split_cases,require_original
from .evaluation import score

CONDITIONS=('fsm','base_slm','slm','hybrid')

class Session:
    def __init__(self,inventory,fsm,slm,gate,heldout,log_path,identity):
        require_original(heldout)
        self.inventory=inventory;self.fsm=fsm;self.slm=slm;self.gate=gate
        self.heldout={c['case_id']:c for c in heldout};self.log_path=Path(log_path);self.identity=identity
        self.reset()
    def reset(self):
        self.history=[];self.pending=None;self.accepted_item=None;self.cart=None
        self.interaction_id=str(uuid.uuid4())
        return {'message':'New interaction. Enter one item request.'}
    def log(self,entry):
        self.log_path.parent.mkdir(parents=True,exist_ok=True)
        entry={'suite':'interactive_only','timestamp':datetime.now(timezone.utc).isoformat(),
               'interaction_id':self.interaction_id,**self.identity,**entry}
        with self.log_path.open('a') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
    def compare(self,inp,case=None):
        outputs={}
        for condition in CONDITIONS:
            self.slm.base_only=condition=='base_slm'
            self.slm.use_validation_selection=condition=='hybrid'
            agent=Agent(self.inventory,self.fsm,self.slm,'slm' if condition=='base_slm' else condition,
                        self.gate['threshold'],self.gate['margin'])
            if case is None:agent.state.item_id=self.accepted_item
            output=agent.reply(inp)
            output['recovery_model']=self.slm.selected_recovery_model if condition=='hybrid' else ('base' if condition=='base_slm' else 'adapter' if condition=='slm' else None)
            outputs[condition]={'output':output,'metrics':score(case,output) if case else None}
        result={'comparison_id':str(uuid.uuid4()),'history':copy.deepcopy(inp['history']),'outputs':outputs,
            'mode':'heldout_replay' if case else 'your_own_request',
            'targets':copy.deepcopy(case['targets']) if case else None,
            'case_id':case['case_id'] if case else None}
        # Targets are attached AFTER all model calls. They never enter prompts.
        self.log({'event':'comparison',**result})
        return result
    def handle(self,request):
        op=request['op']
        if op=='reset':return self.reset()
        if op=='metadata':
            return {**self.identity,'selected_recovery_model':self.slm.selected_recovery_model,
                'heldout_cases':[{'case_id':c['case_id'],'task':c['task'],'cart':c['input']['cart'],
                    'request':next((t['text'] for t in c['input']['history'] if t['role']=='participant'),'')[:100]}
                    for c in self.heldout.values()]}
        if op=='replay':
            case=self.heldout[request['case_id']]
            self.reset();result=self.compare(copy.deepcopy(case['input']),case)
            result['message']='This case was excluded from training and validation for the loaded fold. Replays are logged separately.'
            self.pending=result
            return result
        if op=='message':
            if self.pending and self.pending['mode']=='your_own_request':
                raise ValueError('Choose a reply to continue, or click New interaction.')
            if self.pending:self.reset()
            cart=request['cart'];message=request['text'].strip()
            if not message:raise ValueError('Enter an item request or follow-up.')
            if self.cart is not None and cart!=self.cart:raise ValueError('Click New interaction before changing carts.')
            self.cart=cart
            history=self.history+[{'role':'participant','text':message}]
            inp={'cart':cart,'history':history,'recovery_requested':bool(request.get('recovery_requested',False))}
            result=self.compare(inp);self.history=history;self.pending=result
            return result
        if op=='continue':
            if not self.pending or self.pending['mode']!='your_own_request':raise ValueError('Compare your own request first.')
            output=self.pending['outputs'][request['condition']]['output']
            self.history.append({'role':'robot','text':output['text']});self.accepted_item=output['item_id']
            self.log({'event':'continue','comparison_id':self.pending['comparison_id'],'condition':request['condition']})
            self.pending=None
            return {'history':self.history,'message':'Reply added. Enter a follow-up for this same item, or start a new interaction.'}
        if op=='rate':
            if not self.pending:raise ValueError('Compare replies before saving a review.')
            if request['condition'] not in CONDITIONS:raise ValueError('Unknown system')
            if request['rating'] not in ('appropriate','inappropriate','uncertain'):raise ValueError('Choose a rating')
            self.log({'event':'human_review','comparison_id':self.pending['comparison_id'],
                'condition':request['condition'],'rating':request['rating'],'reason':str(request.get('reason',''))})
            return {'message':'Review saved separately. It does not change model training or routing.'}
        raise ValueError('Unknown tester operation')

def load_session(run,fold,seed,log_path):
    from .slm import HFSLM
    root=Path(__file__).resolve().parents[1];run=Path(run)
    config=read(run/'config.json');plans=read(run/'fold_plan.json')
    cases=read(run/'original_cases.json' if (run/'original_cases.json').exists() else root/'data/original_cases.json')['cases']
    if read(run/'environment.json')['cases_sha256']!=digest(cases):raise ValueError('Run dataset does not match this tester')
    plan=next(p for p in plans if p['fold']==fold);parts=split_cases(cases,plan)
    inv=Inventory(run/'inventory.json' if (run/'inventory.json').exists() else root/'data/inventory.json')
    folder=run/f'seed_{seed}'/f'fold_{fold}'
    slm=HFSLM.from_config(config,folder/'adapter');fsm=FSM().fit([c for c in parts['train'] if c['task']=='retrieval'])
    gate=read(folder/'resolution_gate.json')
    # Warm both real-model paths once; loading and warm-up are reported separately.
    started=time.perf_counter();h=[{'role':'participant','text':'Where is blue tape?'}]
    slm.resolve(h,'lab',inv.table('lab'))
    for base_only in (True,False):
        slm.base_only=base_only;slm.recover(h,'lab',inv.table('lab'))
    slm.base_only=False
    identity={'run':str(run),'fold':fold,'seed':seed,'model':config['model_id'],
              'model_load_seconds':slm.load_seconds,'warmup_seconds':time.perf_counter()-started}
    return Session(inv,fsm,slm,gate,parts['test'],log_path,identity)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--fold',type=int,default=0)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--log',required=True)
    args=parser.parse_args();channel=sys.stdout
    def emit(obj):channel.write(json.dumps(obj,ensure_ascii=False)+'\n');channel.flush()
    try:
        with redirect_stdout(sys.stderr):session=load_session(args.run,args.fold,args.seed,args.log)
        emit({'ok':True,'result':session.handle({'op':'metadata'})})
    except Exception as e:
        emit({'ok':False,'error':str(e)});return
    for line in sys.stdin:
        try:
            with redirect_stdout(sys.stderr):result=session.handle(json.loads(line))
            emit({'ok':True,'result':result})
        except Exception as e:emit({'ok':False,'error':str(e)})

if __name__=='__main__':main()
