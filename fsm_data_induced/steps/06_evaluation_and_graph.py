# @title Evaluation helpers and graph export
def evaluate_requests(model,examples):
    rows=[]
    for e in examples:
        result=model.request(e['query'],e['setting'])
        seen=any(r['setting']==e['setting'] and r['item_key']==e['item_key'] for r in model.table)
        correct=result.get('item_key')==e['item_key'] and result.get('drawer')==e['drawer']
        rows.append({'trial_id':e['trial_id'],'query':e['query'],'setting':e['setting'],'seen_item':seen,
                     'expected_item_from_text':e['item_key'],'expected_drawer_from_text':e['drawer'],
                     'correct':correct,'prediction':result})
    returned=[r for r in rows if r['prediction']['action']=='RETURN_LOCATION']
    seen=[r for r in rows if r['seen_item']]
    return {'cases':len(rows),'correct':sum(r['correct'] for r in rows),
            'accuracy_all':sum(r['correct'] for r in rows)/len(rows) if rows else None,
            'answered':len(returned),'answer_precision':sum(r['correct'] for r in returned)/len(returned) if returned else None,
            'seen_item_cases':len(seen),'seen_item_accuracy':sum(r['correct'] for r in seen)/len(seen) if seen else None,
            'unseen_item_cases':len(rows)-len(seen),'rows':rows}

def tune_on_validation(model,examples):
    # Never receives the reference inventory or held-out test requests.
    default=model.thresholds
    if not examples:return {'status':'no_validation_examples','chosen':asdict(default)}
    grid=[.8,.85,.9] if model.embeddings.backend=='bge' else [.55,.65,.75]
    reports=[]
    for frame in grid:
        model.thresholds=Thresholds(frame,default.item,default.margin)
        report=evaluate_requests(model,examples)
        wrong=report['answered']-sum(r['correct'] for r in report['rows'] if r['prediction']['action']=='RETURN_LOCATION')
        reports.append({'frame':frame,'correct':report['correct'],'wrong_answers':wrong,'cases':report['cases']})
    winner=max(reports,key=lambda r:(r['correct']-3*r['wrong_answers'],r['frame']))
    model.thresholds=Thresholds(winner['frame'],default.item,default.margin)
    return {'chosen':asdict(model.thresholds),'grid':reports,'note':'Frame threshold selected on validation; item/margin values remain explicit defaults.'}

def save_graph(model,path):
    from graphviz import Digraph, escape
    g=Digraph('learned_request_acceptor');g.attr(rankdir='LR',label='Learned request FSM',labelloc='t',fontname='Helvetica',fontsize='20',pad='.3',nodesep='.7',ranksep='1')
    g.attr('node',fontname='Helvetica',fontsize='13',style='filled',fillcolor='#eaf3fa',color='#31556e')
    g.attr('edge',fontname='Helvetica',fontsize='10',color='#64748b')
    g.node('start',shape='point');g.edge('start',model.machine.initial_state.state_id)
    for s in model.machine.states:g.node(s.state_id,shape='doublecircle' if s.output else 'circle')
    for s in model.machine.states:
        by_target=defaultdict(list)
        for word,target in s.transitions.items():by_target[target.state_id].append(word)
        for target,words in by_target.items():
            words=sorted(words)
            label='\n'.join(' / '.join(words[i:i+6]) for i in range(0,len(words),6))
            g.edge(s.state_id,target,label=escape(label))
    path=Path(path);path.write_text(g.source)
    return g