import copy
import json
import math
import random
import statistics
from collections import Counter,defaultdict
from pathlib import Path
from .core import Agent,FSM,Inventory,STRATEGIES,drawer_mentions,digest
from .data import read,write,require_original,fold_plan,split_cases,make_test_n

def score(case,output):
    result={'acceptable_action':output['action'] in case['targets']['acceptable_actions']}
    if case['task']=='retrieval':
        expected=case['targets'];ds=drawer_mentions(output['text'])
        affirmative=not ('?' in output['text'] or __import__('re').search(r'\b(not|never|maybe|perhaps)\b',output['text'],__import__('re').I))
        result.update(joint_item_drawer_correct=bool(output['action']=='locate' and output['item_id']==expected['item_id']
          and output['drawer']==expected['drawer'] and ds and all(x==expected['drawer'] for x in ds) and affirmative),
          item_correct=output['item_id']==expected['item_id'],
          wrong_drawer=bool(ds and any(x!=expected['drawer'] for x in ds)),
          answered=output['action']=='locate',clarified=output['action']=='clarify',
          raw_resolution_correct=None if output.get('proposal') is None else output['proposal'].get('item_id')==expected['item_id'])
    else:
        prefs=set(case['targets']['preferences'])
        result.update(preference_agreement=output['strategy'] in prefs,
          raw_preference_agreement=output['raw_strategy'] in prefs,
          raw_strategy_delivered=output['strategy']==output['raw_strategy'])
    return result

def percentile(values,p):
    if not values:return None
    a=sorted(values);x=(len(a)-1)*p;lo=int(x);hi=math.ceil(x)
    return a[lo]+(a[hi]-a[lo])*(x-lo)

def mean(values):return sum(values)/len(values) if values else None

def clustered_ci(rows,metric,seed=4,n=1000):
    by_pid=defaultdict(list)
    for r in rows:
        if r['metrics'].get(metric) is not None:by_pid[r['participant_id']].append(float(r['metrics'][metric]))
    if len(by_pid)<2:return None
    rng=random.Random(seed);pids=sorted(by_pid);samples=[]
    for _ in range(n):
        chosen=[rng.choice(pids) for __ in pids]
        samples.append(mean([v for p in chosen for v in by_pid[p]]))
    return [percentile(samples,.025),percentile(samples,.975)]

def summarize(predictions,suite):
    if not predictions:return {'suite':suite,'conditions':{}}
    if any(r['suite']!=suite for r in predictions):raise ValueError('Never pool original results with Test N')
    grouped=defaultdict(list)
    for p in predictions:grouped[(p['condition'],p['task'])].append(p)
    result={}
    for (condition,task),rows in grouped.items():
        metrics={k:mean([float(r['metrics'][k]) for r in rows if r['metrics'].get(k) is not None]) for k in set().union(*(r['metrics'] for r in rows))}
        primary='joint_item_drawer_correct' if task=='retrieval' else 'preference_agreement'
        metrics.update(n_cases=len(rows),n_participants=len({r['participant_id'] for r in rows}),
          primary_95_ci=clustered_ci(rows,primary),slm_invocation_rate=mean([r['output']['route']=='slm' for r in rows]),
          median_seconds=percentile([r['output']['seconds'] for r in rows],.5),p95_seconds=percentile([r['output']['seconds'] for r in rows],.95),
          mean_forward_calls=mean([r['output']['forward_calls'] for r in rows]),
          mean_prompt_tokens=mean([r['output']['prompt_tokens'] for r in rows]))
        metrics['latency_by_route']={route:{
            'n_cases':sum(r['output']['route']==route for r in rows),
            'median_seconds':percentile([r['output']['seconds'] for r in rows if r['output']['route']==route],.5),
            'p95_seconds':percentile([r['output']['seconds'] for r in rows if r['output']['route']==route],.95)}
            for route in sorted({r['output']['route'] for r in rows})}
        if task=='retrieval':
            metrics['accuracy_among_answered']=mean([r['metrics']['joint_item_drawer_correct'] for r in rows if r['metrics']['answered']])
        if task=='recovery':
            metrics['by_failure_type']={t:mean([r['metrics']['preference_agreement'] for r in rows if t in r['failure_types']])
              for t in sorted({t for r in rows for t in r['failure_types']})}
            metrics['strategy_counts']=dict(Counter(r['output']['strategy'] for r in rows))
            metrics['largest_strategy_fraction']=max(metrics['strategy_counts'].values())/len(rows)
        result.setdefault(condition,{})[task]=metrics
    return {'suite':suite,'status':'development comparison on original participants previously evaluated in v4; provisional text review','conditions':result,
            'latency':'fresh end-to-end executions, model loading excluded; no cached latency substitution',
            'recovery_scope':'selection at annotated failure points; not live failure detection or physical recovery success'}

def run_cases(cases,inventory,fsm,condition,slm=None,threshold=.98,margin=.8,suite='original',fold=None):
    if suite=='original':require_original(cases)
    elif any(c['suite']!=suite for c in cases):raise ValueError('Wrong test suite')
    agent=Agent(inventory,fsm,slm,condition,threshold,margin);rows=[]
    # Separate warm-up; discarded rather than contaminating latency quantiles.
    warmed_tasks=set()
    for c in cases:
        if c['task'] not in warmed_tasks:
            warm=copy.deepcopy(c['input']);agent.reply(warm);agent.reset()
            warmed_tasks.add(c['task'])
    for c in cases:
        agent.reset();output=agent.reply(c['input'])
        rows.append({'case_id':c['case_id'],'trial_id':c['trial_id'],'participant_id':c['participant_id'],
          'suite':suite,'task':c['task'],'condition':condition,'fold':fold,'output':output,'metrics':score(c,output),
          'failure_types':c['targets'].get('failure_types',[])})
    return rows

def prior_rows(train,test,oracle=False):
    import time
    from .core import render
    require_original(train);require_original(test)
    counts=Counter(s for c in train if c['task']=='recovery' for s in c['targets']['preferences'])
    conditional=defaultdict(Counter)
    for c in train:
        if c['task']=='recovery':
            for t in c['targets']['failure_types']:conditional[t].update(c['targets']['preferences'])
    rows=[]
    for c in test:
        if c['task']!='recovery':continue
        start=time.perf_counter();scores=counts
        if oracle:
            scores=Counter()
            for t in c['targets']['failure_types']:scores.update(conditional[t])
        strategy=max(STRATEGIES,key=lambda s:scores[s])
        out={'action':'recover','strategy':strategy,'raw_strategy':strategy,'item_id':None,'drawer':None,
             'text':render('recover',strategy=strategy),'route':'prior','seconds':time.perf_counter()-start,
             'forward_calls':0,'prompt_tokens':0,'generated_tokens':0,'decision_tokens':0}
        rows.append({'case_id':c['case_id'],'trial_id':c['trial_id'],'participant_id':c['participant_id'],
           'suite':'original','task':'recovery','condition':'oracle_gold_failure_type_prior' if oracle else 'training_frequency_prior',
           'output':out,'metrics':score(c,out),'failure_types':c['targets']['failure_types']})
    return rows

def calibrate(slm,validation,inventory):
    require_original(validation)
    observations=[]
    for c in validation:
        if c['task']!='retrieval':continue
        i=c['input'];p=slm.resolve(i['history'],i['cart'],inventory.table(i['cart']))
        observations.append((p,p.get('item_id')==c['targets']['item_id']))
    candidates=[]
    for threshold in (.9,.95,.98,.995):
        for margin in (.2,.5,.8):
            accepted=[correct for p,correct in observations if p.get('item_id') and p['probability']>=threshold and p['margin']>=margin]
            if len(accepted)>=3 and all(accepted):candidates.append((len(accepted),threshold,margin))
    if candidates:
        _,threshold,margin=max(candidates,key=lambda x:(x[0],x[1],x[2]))
    else:threshold,margin=1.01,1.01
    return {'threshold':threshold,'margin':margin,'n_validation_requests':len(observations),
       'rule':'zero observed validation errors with >=3 accepted; otherwise abstain on SLM item proposals',
       'scope':'conditional choice gate; not evidence of calibration on unknown items or Test N'}

def evaluate_baselines(cases,inventory,out,plans=None):
    require_original(cases);plans=plans or fold_plan(cases);predictions=[]
    for plan in plans:
        parts=split_cases(cases,plan);fsm=FSM().fit([c for c in parts['train'] if c['task']=='retrieval'])
        for condition in ('fsm','semantic_fsm'):
            predictions+=run_cases(parts['test'],inventory,fsm,condition,fold=plan['fold'])
        predictions+=prior_rows(parts['train'],parts['test'])
        predictions+=prior_rows(parts['train'],parts['test'],oracle=True)
    write(Path(out)/'original_predictions.json',predictions)
    summary=summarize(predictions,'original');write(Path(out)/'original_summary.json',summary);return summary

def full_experiment(cases,inventory,out,config,folds=None,seeds=(42,),test_n=False):
    import gc,torch
    from .slm import train_recovery,HFSLM
    require_original(cases);plans=fold_plan(cases,seed=config['split_seed']);out=Path(out)
    if folds is not None and (not folds or len(folds)!=len(set(folds)) or not set(folds)<=set(range(len(plans)))):
        raise ValueError('Choose one or more unique valid fold numbers')
    if not seeds or len(seeds)!=len(set(seeds)):raise ValueError('Choose one or more unique training seeds')
    if out.exists():raise ValueError('Use a new output directory; existing results are not overwritten')
    out.mkdir(parents=True);write(out/'fold_plan.json',plans);write(out/'config.json',config)
    write(out/'inventory.json',{'items':inventory.items});write(out/'original_cases.json',{'cases':cases})
    write(out/'environment.json',{'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cases_sha256':digest(cases)})
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required; no trained model or SLM scores were produced')
    for seed in seeds:
        predictions=[];noisy=[]
        for plan in plans:
            if folds is not None and plan['fold'] not in folds:continue
            parts=split_cases(cases,plan);fold=plan['fold'];folder=out/f'seed_{seed}'/f'fold_{fold}'
            print(f"\nFOLD {fold+1}/5 — new recovery adapter, seed {seed}. "
                  f"Train: {sum(c['task']=='recovery' for c in parts['train'])}; "
                  f"validation: {sum(c['task']=='recovery' for c in parts['validation'])}; "
                  f"test: {sum(c['task']=='recovery' for c in parts['test'])} original recovery cases.",flush=True)
            adapter=train_recovery(parts['train'],parts['validation'],inventory,folder,config,seed)
            slm=HFSLM.from_config(config,adapter);fsm=FSM().fit([c for c in parts['train'] if c['task']=='retrieval'])
            write(folder/'inference_load.json',{'model_load_seconds':slm.load_seconds,'gpu':torch.cuda.get_device_name(0)})
            gate=calibrate(slm,parts['validation'],inventory);write(folder/'resolution_gate.json',gate)
            noisy_cases=make_test_n(parts['test']) if test_n else []
            if test_n:write(folder/'test_n_cases.json',noisy_cases)
            predictions+=prior_rows(parts['train'],parts['test'])
            predictions+=prior_rows(parts['train'],parts['test'],oracle=True)
            for condition in ('fsm','semantic_fsm','base_slm','slm','selected_slm','hybrid'):
                slm.base_only=condition=='base_slm'
                slm.use_validation_selection=condition in ('selected_slm','hybrid')
                actual='slm' if condition in ('base_slm','selected_slm') else condition
                print(f'Evaluating {condition} on original held-out cases.',flush=True)
                pp=run_cases(parts['test'],inventory,fsm,actual,slm,gate['threshold'],gate['margin'],fold=fold)
                for p in pp:
                    p['condition']=condition
                    p['recovery_model']=slm.selected_recovery_model if slm.use_validation_selection else ('base' if slm.base_only or condition in ('fsm','semantic_fsm') else 'adapter')
                predictions+=pp
                if test_n:
                    nn=run_cases(noisy_cases,inventory,fsm,actual,slm,gate['threshold'],gate['margin'],suite='test_n',fold=fold)
                    for p in nn:p['condition']=condition
                    noisy+=nn
            write(out/f'seed_{seed}'/'original_predictions.json',predictions)
            write(out/f'seed_{seed}'/'original_summary.json',summarize(predictions,'original'))
            if test_n:
                write(out/f'seed_{seed}'/'test_n_predictions.json',noisy)
                write(out/f'seed_{seed}'/'test_n_summary.json',summarize(noisy,'test_n'))
            del slm;gc.collect();torch.cuda.empty_cache()
            print(f'Fold {fold+1}/5 saved to {folder}.',flush=True)
    return out
