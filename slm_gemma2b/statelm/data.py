"""Prepare original-only decision prefixes with explicit source/label separation."""
import copy
import json
import random
import re
from collections import Counter,defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from .core import Inventory,digest,norm

def read(path):return json.loads(Path(path).read_text())
def write(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def require_original(cases):
    if any(c.get('suite')!='original' or c.get('synthetic',False) for c in cases):
        raise ValueError('Synthetic/Test N cases cannot enter original-data training or scoring')

def token(s):
    return {'1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','too':'two'}.get(s.lower(),s.lower())

def source_turns(version):
    """Use actual source text, not edited ASR words in older role annotations.

    Alignment can establish text span, not speaker identity. Unknown/mixed roles
    and low coverage remain exclusions. No noise, synonym or typo is injected.
    """
    segments=version['source_segments'];sw=[];positions=[]
    for j,s in enumerate(segments):
        for m in re.finditer(r'[A-Za-z0-9]+',s['text']):sw.append(token(m[0]));positions.append((j,m.start(),m.end()))
    tw=[];owners=[]
    for i,t in enumerate(version['reviewed_turns']):
        for m in re.finditer(r'[A-Za-z0-9]+',t['text']):tw.append(token(m[0]));owners.append(i)
    matches=defaultdict(list)
    for b in SequenceMatcher(None,tw,sw,autojunk=False).get_matching_blocks():
        for k in range(b.size):matches[owners[b.a+k]].append(b.b+k)
    counts=Counter(owners);out=[];last=-1
    for i,t in enumerate(version['reviewed_turns']):
        mm=matches[i];flags=list(t.get('review_flags',[]))
        if not mm or len(mm)/max(counts[i],1)<.8:flags.append('source_alignment_insufficient')
        if mm:
            a,z=min(mm),max(mm);j0,c0,_=positions[a];j1,_,c1=positions[z]
            while c1<len(segments[j1]['text']) and not segments[j1]['text'][c1].isalnum():c1+=1
            pieces=[segments[j]['text'][c0 if j==j0 else 0:c1 if j==j1 else len(segments[j]['text'])] for j in range(j0,j1+1)]
            text=' '.join(pieces).strip()
            original_neg={x for x in re.findall(r'[a-z]+',t['text'].lower()) if x in ('not','never')}
            source_neg={x for x in re.findall(r'[a-z]+',text.lower()) if x in ('not','never')}
            if original_neg!=source_neg:flags.append('source_polarity_disagreement')
            if a<last:flags.append('source_turn_order_disagreement')
            last=z
            start,end=segments[j0]['start'],segments[j1]['end']
        else:text='';start=end=None
        out.append({'role':t['role'],'text':text,'start':start,'end':end,'source_turn_index':i,
                    'source_version':version['sample_id'],'flags':flags,'role_status':t['review_status']})
    return out

def clean_prefix(turns,end):
    prefix=[]
    for t in turns[:end+1]:
        if 'subsequent_trial_or_setup_exclude_from_current_episode' in t['flags']:
            return None,'crosses_next_trial'
        if t['role']=='operator':continue
        if t['role'] is None or t['flags']:return None,'unresolved_role_or_source_span'
        if t['role'] not in ('participant','robot'):return None,'invalid_speaker'
        if not t['text'].strip():return None,'empty_source_turn'
        u={k:t[k] for k in ('role','text','start','end')}
        if prefix and prefix[-1]['role']==u['role'] and prefix[-1]['start']==u['start'] and prefix[-1]['end']==u['end']:
            prefix[-1]['text']+=' '+u['text']
        else:prefix.append(u)
    if not any(x['role']=='participant' for x in prefix):return None,'no_participant_request'
    return prefix,None

def prepare(source_path,output):
    source=read(source_path);inventory=Inventory(source['inventory']['items']);cases=[];coverage=[]
    evidence={e['sample_id']:(i['item_id'],i['drawer']) for i in source['inventory']['items'] for e in i.get('success_transcript_evidence',[])}
    for trial in source['trials']:
        ann=trial['annotations_not_model_input'];cart=trial['cart']['value'];tid=trial['trial_id'];attempts=[]
        def case(task,history,targets,version,cut):
            inp={'cart':cart,'history':[{'role':x['role'],'text':x['text']} for x in history],
                 'recovery_requested':task=='recovery'}
            # Only past observed timing, no gold failure duration or end time.
            if task=='recovery':
                robot=history[-1];users=[x for x in history[:-1] if x['role']=='participant']
                if robot['role']=='robot' and users and robot['start'] is not None and users[-1]['end'] is not None and robot['start']>=users[-1]['end']:
                    inp['observed_wait_seconds']=robot['start']-users[-1]['end']
            return {'case_id':tid+':'+task,'trial_id':tid,'participant_id':trial['participant_id'],
                'suite':'original','synthetic':False,'task':task,'input':inp,'targets':targets,
                'provenance':{'sample_aliases':trial['sample_aliases'],'version':version['sample_id'],
                  'cut_after_source_turn':cut,'source_transcript_sha256':digest(version['raw_transcript']),
                  'source_text_policy':'source spans; no synthetic corruption or spelling repair',
                  'annotation_status':'provisional_text_review','cart_status':trial['cart']['status'],
                  'recovery_trigger':'annotated_failure_point' if task=='recovery' else None}}
        # Prefer reviewed copies, but do not combine words from different versions.
        versions=sorted(trial['transcript_versions'],key=lambda v:(not v['role_review_status'].startswith('text_review_complete'),v['sample_id']))
        if 'success' in ann['trial_labels']:
            gold={evidence[s] for s in trial['sample_aliases'] if s in evidence}
            if len(gold)!=1:attempts.append(('retrieval','no_independent_clear_item_and_drawer_evidence'))
            else:
                iid,drawer=next(iter(gold));selected=None
                for v in versions:
                    tt=source_turns(v)
                    first_robot=next((j for j,t in enumerate(tt) if t['role']=='robot'),len(tt))
                    users=[j for j in range(first_robot) if tt[j]['role']=='participant']
                    if not users:continue
                    # Eligibility never depends on whether the runtime item matcher succeeds.
                    j=users[-1];h,why=clean_prefix(tt,j)
                    if h:selected=case('retrieval',h,{'item_id':iid,'drawer':drawer,'acceptable_actions':['locate']},v,j)
                    if selected:break
                if selected:cases.append(selected);attempts.append(('retrieval','included'))
                else:attempts.append(('retrieval','request_prefix_needs_review'))
        if any(l.endswith('_failure') for l in ann['trial_labels']):
            prefs=ann['recovery_preferences']['selected_strategies'];selected=None
            if not prefs:attempts.append(('recovery','missing_named_preferences'))
            else:
                for v in versions:
                    tt=source_turns(v);intervals=ann['failure_intervals']
                    for j,t in enumerate(tt):
                        if t['role']!='robot' or t['start'] is None:continue
                        if not any(t['end']>x['clip_start'] and t['start']<x['clip_end'] for x in intervals):continue
                        end=j
                        # Include the continuation of the SAME failure utterance,
                        # never a later correct drawer response just because its
                        # speaker is also robot.
                        while end+1<len(tt) and tt[end+1]['role']=='robot' and re.fullmatch(r'please repeat your (request|last words)',norm(tt[end+1]['text'])):
                            end+=1
                        h,why=clean_prefix(tt,end)
                        if h:
                            selected=case('recovery',h,{'preferences':prefs,'failure_types':[l for l in ann['trial_labels'] if l.endswith('_failure')],
                                                      'acceptable_actions':['recover']},v,end);break
                    if selected:break
                if selected:cases.append(selected);attempts.append(('recovery','included'))
                else:attempts.append(('recovery','no_usable_failure_prefix'))
        coverage.append({'trial_id':tid,'sample_aliases':trial['sample_aliases'],'decisions':attempts})
    require_original(cases)
    result={'schema':1,'suite':'original','status':'provisional_text_review_not_paper_results','source_sha256':digest(source),
        'cases':cases,'coverage':coverage,'counts':dict(Counter(x['task'] for x in cases)),
        'exclusion_counts':dict(Counter(task+':'+reason for x in coverage for task,reason in x['decisions'] if reason!='included'))}
    write(output,result);return result

def fold_plan(cases,folds=5,seed=42):
    require_original(cases)
    # Union copied whole transcripts, not ordinary shared wording such as
    # "Where are the scissors?" across independent participants.
    pids=sorted({str(c['participant_id']) for c in cases});parent={p:p for p in pids}
    def find(x):
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    signatures={}
    for c in cases:
        sig=c['provenance'].get('source_transcript_sha256',c['trial_id'])
        p=str(c['participant_id'])
        if sig in signatures:parent[find(p)]=find(signatures[sig])
        else:signatures[sig]=p
    groups=defaultdict(list)
    for p in pids:groups[find(p)].append(p)
    units=list(groups.values());random.Random(seed).shuffle(units);units.sort(key=len,reverse=True)
    bins=[[] for _ in range(folds)]
    for unit in units:bins[min(range(folds),key=lambda i:len(bins[i]))].extend(unit)
    if any(not b for b in bins):raise ValueError('Not enough independent participant groups')
    return [{'fold':f,'test_participants':sorted(bins[f]),'validation_participants':sorted(bins[(f+1)%folds]),
        'train_participants':sorted(p for j,b in enumerate(bins) if j not in (f,(f+1)%folds) for p in b)} for f in range(folds)]

def split_cases(cases,plan):
    require_original(cases)
    result={s:[c for c in cases if str(c['participant_id']) in plan[s+'_participants']] for s in ('train','validation','test')}
    sets=[{c['trial_id'] for c in result[s]} for s in result]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('Physical trial leaked between partitions')
    return result

def make_test_n(test_cases,seed=123):
    """Text-only spelling perturbations; NOT a realistic accent/audio benchmark.

    Called after splitting. Never used to fit aliases, adapters or thresholds.
    """
    require_original(test_cases);out=[];rng=random.Random(seed)
    for c in test_cases:
        if c['task']!='retrieval':continue
        text=c['input']['history'][-1]['text'];words=list(re.finditer(r'[A-Za-z]{5,}',text))
        if not words:continue
        w=words[-1];offset=rng.randrange(1,len(w[0])-2);s=w[0]
        changed=s[:offset]+s[offset+1]+s[offset]+s[offset+2:]
        if changed==s:continue
        n=copy.deepcopy(c);n.update(case_id=c['case_id']+':test_n',suite='test_n',synthetic=True)
        n['input']['history'][-1]['text']=text[:w.start()]+changed+text[w.end():]
        n['provenance']={'parent_case_id':c['case_id'],'distortion':'internal_adjacent_letter_swap','seed':seed,
            'interpretation':'synthetic text noise; not validated accent simulation','original_text':text}
        out.append(n)
    return out
