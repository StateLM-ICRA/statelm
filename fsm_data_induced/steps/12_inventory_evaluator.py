"""Evaluation only. Reference items never enter induction or query matching."""
from collections import defaultdict, Counter
import hashlib, json

def compare_inventory(table,reference,extractor):
    index=defaultdict(set);gold={}
    for r in reference['items']:
        rid=(r['setting'],r['item_id']);gold[rid]=r
        for name in [r['name'],*r['aliases']]:index[r['setting'],extractor.key(name)].add(rid)
    checked=[];matched=set();correct_ids=set()
    for row in table:
        ids=set().union(*(index[row['setting'],extractor.key(name)] for name in row['aliases']))
        base={'setting':row['setting'],'learned_item':row['item'],'learned_drawer':row['drawer'],
              'source_trials':[e['trial_id'] for e in row['evidence']]}
        if len(ids)!=1:
            checked.append({**base,'status':'unmapped_name' if not ids else 'ambiguous_reference_name',
                            'reference_item':None,'reference_drawer':None});continue
        rid=next(iter(ids));matched.add(rid);target=gold[rid]
        correct=row['drawer']==target['drawer']
        if correct:correct_ids.add(rid)
        checked.append({**base,'status':'correct' if correct else 'drawer_mismatch',
                        'reference_item':target['name'],'reference_drawer':target['drawer']})
    counts=Counter(r['status'] for r in checked)
    return {'learned_entries':len(table),'reference_entries':len(gold),'status_counts':dict(counts),
       'matched_reference_items':len(matched),'correct_reference_items':len(correct_ids),
       'correct_reference_coverage':len(correct_ids)/len(gold) if gold else None,
       'accuracy_among_mapped_entries':counts['correct']/(counts['correct']+counts['drawer_mismatch']) if counts['correct']+counts['drawer_mismatch'] else None,
       'rows':checked,'missing_reference_items':[{'setting':g['setting'],'item':g['name'],'drawer':g['drawer']} for rid,g in gold.items() if rid not in matched],
       'interpretation':'Mapping uses reference aliases only in evaluation; unmapped names are not repaired or merged back into the learned table.'}

def fingerprint(model):
    return hashlib.sha256(json.dumps(model.artifact(),sort_keys=True).encode()).hexdigest()