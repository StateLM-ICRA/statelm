# @title Runtime and explicit manual clarification
class InducedLocator:
    def __init__(self,examples,extractor,backend,sparse_module):
        if not examples:raise ValueError('No accepted training examples; inspect extraction_audit.json')
        self.examples=examples;self.extractor=extractor
        self.table,self.conflicts=build_table(examples)
        self.frames=sorted({e['frame'] for e in examples})
        self.machine,self.graph_stats=learn_acceptor(self.frames)
        # Frequency selection over observed masked robot replies, not a supplied answer.
        template_groups=defaultdict(list)
        for e in examples:template_groups[e['template'].casefold()].append(e)
        winner=max(template_groups,key=lambda t:(len(template_groups[t]),-len(t),t))
        self.template=template_groups[winner][0]['template']
        self.template_evidence=[e['trial_id'] for e in template_groups[winner]]
        self.embeddings=Embeddings(backend,self.frames+[a for r in self.table for a in r['aliases']],sparse_module)
        self.sparse_items=Embeddings('tfidf',[a for r in self.table for a in r['aliases']],sparse_module)
        self.thresholds=Thresholds(frame=.85 if backend=='bge' else .65,item=.82 if backend=='bge' else .55)
        self.embeddings.vectors(self.frames+[r['item'] for r in self.table])

    def artifact(self):
        return {'method':'success-text extraction + induced request acceptor + embedding abstraction + learned table',
          'embedding_backend':self.embeddings.backend,'thresholds':asdict(self.thresholds),
          'template':self.template,'template_evidence':self.template_evidence,
          'request_prototypes':self.frames,'item_tfidf_idf':self.sparse_items.idf,
          'training_trials':sorted({e['trial_id'] for e in self.examples}),
          'training_participants':sorted({e['participant_id'] for e in self.examples}),
          'table':self.table,'graph_stats':self.graph_stats,
          'initial_state':self.machine.initial_state.state_id,
          'states':[{'id':s.state_id,'accepting':bool(s.output),
                     'transitions':{a:t.state_id for a,t in sorted(s.transitions.items())}} for s in self.machine.states]}

    def render(self,row,frame_info=None):
        if row['drawer'] is None:
            return {'action':'LOCATION_CONFLICT','text':'The observed locations disagree; this item needs review.','item':row['item'],'drawer':None}
        return {'action':'RETURN_LOCATION','text':self.template.format(drawer=row['drawer']),
                'item':row['item'],'item_key':row['item_key'],'drawer':row['drawer'],
                'evidence_trials':[e['trial_id'] for e in row['evidence']],**(frame_info or {})}

    def request(self,text,setting):
        if setting not in {'lab','hospital'}:return {'action':'SETTING_REQUIRED','text':'Select lab or hospital.','drawer':None}
        phrases=list(self.extractor.candidates(text))
        # A standalone noun phrase may be misparsed as a verb; exact observed aliases are safe candidates.
        for row in self.table:
            if row['setting']!=setting:continue
            for alias in row['aliases']:
                for m in re.finditer(r'(?<!\w)'+re.escape(alias)+r'(?!\w)',text,re.I):
                    phrases.append({'text':m[0],'key':row['item_key'],'start':m.start(),'end':m.end(),'head':row['head']})
        if not phrases:return {'action':'UNSUPPORTED_REQUEST','text':'I could not recognize an item request.','drawer':None}
        scored=[]
        for phrase in phrases:
            frame=self.extractor.frame(text,phrase)
            if frame in self.frames:idx=self.frames.index(frame);score=1.
            else:
                scores=self.embeddings.scores(frame,self.frames);idx=int(np.argmax(scores));score=scores[idx]
            scored.append((score,len(tokens(phrase['text'])),phrase,frame,self.frames[idx]))
        score,_,phrase,frame,prototype=max(scored,key=lambda x:(x[0],x[1]))
        if score<self.thresholds.frame:
            return {'action':'UNSUPPORTED_REQUEST','text':'That wording is outside the learned request patterns.','drawer':None,'frame_score':score}
        accepted,path=traverse(self.machine,prototype)
        if not accepted:raise AssertionError('A training prototype was rejected by the learned automaton')
        info={'frame_score':score,'request_frame':frame,'matched_frame':prototype,'fsm_path':path}
        scoped=[r for r in self.table if r['setting']==setting]
        exact=[r for r in scoped if r['item_key']==phrase['key']]
        if exact:return self.render(exact[0],{**info,'item_match':'exact_lemma'})
        # Broad family names refer only to items actually learned in this setting.
        family=[r for r in scoped if set(phrase['key'].split())<=set(r['item_key'].split())]
        if len(family)>1:
            return {'action':'ASK_WHICH_ONE','text':'Which item: '+', '.join(r['item'] for r in family)+'?',
                    'options':[r['item_key'] for r in family],'drawer':None,**info}
        if len(family)==1:return self.render(family[0],{**info,'item_match':'unique_learned_family'})
        if not scoped:return {'action':'ITEM_NOT_LEARNED','text':'No location was learned for that setting.','drawer':None,**info}
        semantic=self.embeddings.scores(phrase['text'],[r['item'] for r in scoped])
        sparse=self.sparse_items.scores(phrase['text'],[r['item'] for r in scoped])
        # Scale each method relative to its declared acceptance threshold.
        sims=[max(s/self.thresholds.item,t/.55) for s,t in zip(semantic,sparse)]
        order=sorted(range(len(scoped)),key=lambda i:sims[i],reverse=True);best=order[0]
        margin=sims[best]-(sims[order[1]] if len(order)>1 else 0.)
        # Semantic proximity alone must not equate different objects (e.g. pliers and scissors).
        lexical=SequenceMatcher(None,phrase['key'],scoped[best]['item_key']).ratio()
        modifier_conflict=phrase['head']==scoped[best]['head'] and phrase['key']!=scoped[best]['item_key']
        if sims[best]>=1. and margin>=self.thresholds.margin and lexical>=.72 and not modifier_conflict:
            return self.render(scoped[best],{**info,'item_match':'embedding_with_lexical_guard','semantic_score':semantic[best],
                              'tfidf_score':sparse[best],'normalized_item_margin':margin})
        return {'action':'ITEM_NOT_LEARNED','text':'I have no learned location for that item.','drawer':None,**info}

class Session:
    """Explicitly hand-designed clarification/missing-item extension, not induced policy."""
    def __init__(self,model,setting='lab'):self.model=model;self.setting=setting;self.pending=[]
    def ask(self,text):
        if self.pending:
            # A complete new question cancels old candidates; a short descriptor narrows them.
            if re.search(r'\b(where|find|locate|need|want|get|give|show)\b',text,re.I):self.pending=[]
            else:
                words=set(tokens(text))-{'the','one','please'}
                candidates=[r for r in self.pending if words and words<=set(tokens(r['item']))]
                if len(candidates)==1:
                    self.pending=[];return self.model.render(candidates[0],{'extension':'manual_clarification'})
                if candidates:self.pending=candidates
                return {'action':'ASK_WHICH_ONE','text':'Please choose: '+', '.join(r['item'] for r in self.pending)+'.','drawer':None}
        result=self.model.request(text,self.setting)
        if result['action']=='ASK_WHICH_ONE':
            self.pending=[r for r in self.model.table if r['setting']==self.setting and r['item_key'] in result['options']]
        return result