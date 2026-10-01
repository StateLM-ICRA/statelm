from __future__ import annotations
import copy
import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

STRATEGIES = ('clarifying_prompt', 'self_correction', 'specific_redirection', 'confidence_check',
              'guided_reset', 'transparency_cue', 'partial_understanding_repair', 'multimodal_redundancy')
NUMBERS = {'one':1,'first':1,'two':2,'second':2,'three':3,'third':3,'four':4,'fourth':4,'five':5,'fifth':5,'six':6,'sixth':6}

def select_recovery(scores,voice_and_display_available=False):
    """The same executable decision rule is used in validation and deployment."""
    allowed=[s for s in STRATEGIES if s!='multimodal_redundancy' or voice_and_display_available]
    return max(allowed,key=lambda s:scores[s])

def norm(s):
    return ' '.join(re.findall(r'[a-z0-9]+', s.lower()))

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def drawer_mentions(text):
    tokens = '|'.join(NUMBERS)+r'|\d+'
    matches = re.findall(r'\b(?:drawer\s+('+tokens+r')|('+tokens+r')\s+drawer)\b',text.lower())
    return [NUMBERS.get(x, int(x) if x.isdigit() else None) for pair in matches for x in pair if x]

def rejected(text):
    return bool(re.search(r"\b(not (there|here|in)|wrong|incorrect|isn.t (there|here)|can.t find|cannot find|couldn.t find)\b",text,re.I))

def followup(text):
    return bool(re.fullmatch(r'(which (one|drawer)( (was|is) (it|that))?( again)?|where (is it|are they)|'
                            r'(please )?(repeat|say)( (it|that))?( again)?( please)?|'
                            r'(can|could) you repeat( (it|that))?( please)?)',norm(text)))

def failure_evidence(history, inventory, cart):
    if not history:return False
    if history[-1]['role']=='participant' and rejected(history[-1]['text']):return True
    if history[-1]['role']!='robot':return False
    text=history[-1]['text']
    if re.search(r'not understand|am delayed|repeat your request',text,re.I):return True
    if re.search(r'open the drawer',text,re.I) and not drawer_mentions(text):return True
    item=None
    for t in reversed(history[:-1]):
        if t['role']=='participant':
            hits=inventory.exact(t['text'],cart)
            if len(hits)==1:item=hits[0]
            break
    return bool(item and drawer_mentions(text) and any(x!=item['drawer'] for x in drawer_mentions(text)))

class Inventory:
    def __init__(self, source):
        data=json.loads(Path(source).read_text()) if isinstance(source,(str,Path)) else copy.deepcopy(source)
        self.items=data['items'] if isinstance(data,dict) else data
        keys=[(x['setting'],x['item_id']) for x in self.items]
        if len(keys)!=len(set(keys)):raise ValueError('Duplicate cart/item inventory key')
        if any(type(x['drawer']) is not int or x['drawer'] not in range(1,7 if x['setting']=='hospital' else 6) for x in self.items):
            raise ValueError('Invalid drawer in reference inventory')
        self.base_hash=digest(self.items)
        self.additions=[]

    def get(self,cart,item_id):
        return next((x for x in self.items+self.additions if x['setting']==cart and x['item_id']==item_id),None)

    def table(self,cart):
        return [{'item_id':x['item_id'],'name':x['name'],'drawer':x['drawer']} for x in self.items+self.additions if x['setting']==cart]

    def exact(self,text,cart):
        s=norm(text);hits=[]
        for x in self.items+self.additions:
            if x['setting']!=cart:continue
            for a in set(x.get('aliases',[])+[x['name']]):
                for m in re.finditer(r'(?<!\w)'+re.escape(norm(a))+r'(?!\w)',s):hits.append((m.start(),m.end(),x))
        hits=[h for h in hits if not any(k[0]<=h[0] and k[1]>=h[1] and k[1]-k[0]>h[1]-h[0] for k in hits)]
        return list({x[2]['item_id']:x[2] for x in hits}.values())

    def add_report(self,cart,name,drawer,report,*,enabled=False):
        if not enabled:raise ValueError('Inventory updates are disabled in the original benchmark')
        if cart not in ('lab','hospital') or type(drawer) is not int or drawer not in range(1,7 if cart=='hospital' else 6):
            raise ValueError('Invalid cart/drawer')
        if not name.strip() or not report.strip():raise ValueError('Explicit item name and user location report required')
        if self.exact(name,cart):raise ValueError('Existing inventory entries cannot be overwritten by a report')
        item_id='user_'+hashlib.sha256((cart+norm(name)).encode()).hexdigest()[:12]
        x={'setting':cart,'item_id':item_id,'name':name.strip(),'drawer':drawer,'aliases':[name.strip()],
           'provenance':{'kind':'user_report','report':report,'recorded_at':datetime.now(timezone.utc).isoformat()}}
        self.additions.append(x)
        return x

@dataclass
class State:
    phase:str='ready'
    item_id:str|None=None
    last_action:str|None=None

class FSM:
    """Authored legal transitions; optional success-derived utterance index.

    The index is fitted only on training success requests. It is not falsely
    described as fully inducing a dialogue policy from failed robot utterances.
    """
    TRANSITIONS={p:{'locate':'guided','clarify':'awaiting_item','recover':'recovering',
                    'unknown':'awaiting_item','acknowledge':'done'} for p in
                 ('ready','guided','awaiting_item','recovering','done')}
    def __init__(self):self.examples=[];self.observed_edges=Counter()
    def fit(self,cases):
        self.examples=[];self.observed_edges=Counter()
        for c in cases:
            if c['suite']!='original' or c['task']!='retrieval':raise ValueError('FSM induction accepts original successful retrieval only')
            item=c['targets']['item_id']; text=c['input']['history'][-1]['text']
            self.examples.append({'cart':c['input']['cart'],'text':text,'item_id':item})
            self.observed_edges['ready->guided']+=1
        return self
    def transition(self,state,action,item_id):
        state.phase=self.TRANSITIONS[state.phase][action]
        # Never reuse an old item for a new unresolved request.
        if action in ('locate','clarify','unknown'):state.item_id=item_id
        elif action=='recover' and item_id:state.item_id=item_id
        if action=='acknowledge':state.item_id=None
        state.last_action=action
    @staticmethod
    def grams(text):
        s=' '+norm(text)+' ';return Counter(s[i:i+3] for i in range(max(0,len(s)-2)))
    def semantic(self,text,cart):
        a=self.grams(text);scores={}
        for x in self.examples:
            if x['cart']!=cart:continue
            b=self.grams(x['text']);den=(sum(v*v for v in a.values())*sum(v*v for v in b.values()))**.5
            score=sum(v*b.get(k,0) for k,v in a.items())/den if den else 0
            scores[x['item_id']]=max(scores.get(x['item_id'],0),score)
        ordered=sorted(scores.items(),key=lambda x:(-x[1],x[0]))
        return (ordered[0][0],ordered[0][1],ordered[0][1]-(ordered[1][1] if len(ordered)>1 else 0)) if ordered else (None,0,0)

def location(item):
    plural=item['item_id'] in {'sd_cards','leds','orange_pencils','yellow_pencils','batteries','red_pencils','wires',
        'blue_markers','blue_pencils','gloves','scissors','pliers','wipes','surgical_gloves','face_masks'}
    return f"The {item['name']} {'are' if plural else 'is'} in drawer {item['drawer']}."

def render(action,item=None,strategy=None,location_rejected=False):
    if action=='locate':
        if item is None:raise ValueError('Location action requires an inventory item')
        return location(item)
    if action=='unknown':return "I don't have a location recorded for that item. If you find it, you can tell me its drawer so I can record it."
    if action=='acknowledge':return "You're welcome."
    if action=='clarify':return 'Which item do you mean? Please give its name or a distinguishing detail.'
    if strategy not in STRATEGIES:raise ValueError('Recovery requires a valid model-selected strategy')
    named=f"the {item['name']}" if item else 'the item'
    if strategy=='clarifying_prompt':return 'Was the item missing, or was my instruction unclear?' if item else 'Which item do you need? Please repeat its name or describe it.'
    if strategy=='confidence_check':return f'Do you mean {named}?' if item else 'Could you confirm the name of the item you need?'
    if strategy=='guided_reset':return "Let's start this request again. Which item do you need?"
    if strategy=='transparency_cue':return "Sorry, I haven't provided useful guidance yet. Let me check the request again."
    if strategy=='partial_understanding_repair':return f'I understood that you need {named}. What part of my guidance should I clarify?' if item else 'I understand you need help finding something. What is the item called?'
    if strategy=='multimodal_redundancy':return 'Here are the written instructions: '+(location(item) if item else 'Please enter or say the item name.')
    if location_rejected and item:
        return f"The inventory lists {named} in drawer {item['drawer']}, but you couldn't find it there. Could you describe what you found?"
    if strategy=='self_correction':return 'Sorry that my earlier response did not help. '+(location(item) if item else 'Please repeat the item name so I can check it.')
    return f"For {named}, check the drawer labeled {item['drawer']}." if item else 'Please tell me the item name so I can give a specific direction.'

class Agent:
    def __init__(self,inventory,fsm=None,slm=None,condition='hybrid',resolve_threshold=.98,resolve_margin=.8,semantic_threshold=.97):
        if condition not in ('fsm','semantic_fsm','slm','hybrid'):raise ValueError('Unknown condition')
        self.inventory=inventory;self.fsm=fsm or FSM();self.slm=slm;self.condition=condition
        self.threshold=resolve_threshold;self.margin=resolve_margin;self.semantic_threshold=semantic_threshold
        self.state=State()
    def reset(self):self.state=State()
    def reply(self,input_data):
        start=time.perf_counter()
        # Copy ONLY public input fields. Gold data cannot flow through **kwargs.
        cart=input_data['cart'];history=copy.deepcopy(input_data['history'])
        if cart not in ('lab','hospital'):raise ValueError('Cart must be configured')
        if not history or any(t['role'] not in ('participant','robot') for t in history):raise ValueError('Only reviewed participant/robot prefix is allowed')
        user=next((t['text'] for t in reversed(history) if t['role']=='participant'),'')
        is_recovery=bool(input_data.get('recovery_requested',False)) or failure_evidence(history,self.inventory,cart)
        hits=self.inventory.exact(user,cart)
        item=hits[0] if len(hits)==1 else None
        if not item and (followup(user) or rejected(user)):
            item=self.inventory.get(cart,self.state.item_id)
            if not item:
                for t in reversed(history[:-1]):
                    if t['role']=='participant':
                        hh=self.inventory.exact(t['text'],cart)
                        if hh:item=hh[0] if len(hh)==1 else None;break
        ack=norm(user) in ('thanks','thank you','got it','found it','thank you very much')
        # A negated or multi-item request is never an exact confident resolution.
        ambiguous=(len(hits)>1 or bool(re.search(r'\b(or|not|instead)\b',norm(user)))) and not is_recovery
        if ambiguous:item=None
        action='acknowledge' if ack and not is_recovery else ('locate' if item else 'clarify')
        stats={'forward_calls':0,'prompt_tokens':0,'generated_tokens':0,'decision_tokens':0};proposal=None
        if not item and self.condition=='semantic_fsm' and not is_recovery and not ambiguous:
            iid,p,gap=self.fsm.semantic(user,cart)
            if p>=self.semantic_threshold and gap>=.2:item=self.inventory.get(cart,iid);action='locate'
        use_slm=self.condition=='slm' or (self.condition=='hybrid' and (is_recovery or (not item and not ack)))
        if use_slm and self.slm is None:raise RuntimeError('SLM is required for this route; no silent mock or graph substitution')
        if use_slm and (self.condition=='slm' or not item):
            proposal=self.slm.resolve(history,cart,self.inventory.table(cart))
            for k in stats:stats[k]+=proposal.get(k,0)
            candidate=self.inventory.get(cart,proposal.get('item_id'))
            exact_confirmation=bool(candidate and len(hits)==1 and hits[0]['item_id']==candidate['item_id'])
            exact_conflict=bool(candidate and len(hits)==1 and hits[0]['item_id']!=candidate['item_id'])
            if candidate and not ambiguous and not exact_conflict and (exact_confirmation or (proposal['probability']>=self.threshold and proposal['margin']>=self.margin)):
                item=candidate;action='locate'
            elif not hits and proposal.get('action')=='unknown' and proposal['probability']>=self.threshold and proposal['margin']>=self.margin:
                item=None;action='unknown'
            elif proposal.get('action')=='acknowledge' and ack and not is_recovery:
                item=None;action='acknowledge'
            else:item=None;action='clarify'
        strategy=None;raw_strategy=None
        if is_recovery:
            if use_slm:
                recovery=self.slm.recover(history,cart,self.inventory.table(cart),input_data.get('observed_wait_seconds'))
                for k in stats:stats[k]+=recovery.get(k,0)
                raw_strategy=recovery['strategy']
                strategy=select_recovery(recovery['scores'],input_data.get('voice_and_display_available',False))
                action='recover'
            else:
                # Fixed, competitive no-SLM recovery baseline. The HYBRID never
                # uses this branch: its recovery choice always comes from SLM.
                action='recover';strategy=raw_strategy='clarifying_prompt'
        text=render(action,item,strategy,rejected(user))
        self.fsm.transition(self.state,action,item['item_id'] if item else None)
        return {'action':action,'item_id':item['item_id'] if item else None,
                'drawer':item['drawer'] if item and drawer_mentions(text) else None,
                'strategy':strategy,'raw_strategy':raw_strategy,'text':text,
                'route':'slm' if use_slm else 'fsm','proposal':proposal,
                'state':asdict(self.state),'seconds':time.perf_counter()-start,**stats}
