# @title Table induction and embeddings
def build_table(examples):
    grouped=defaultdict(list)
    for e in examples:grouped[e['setting'],e['item_key']].append(e)
    table=[];conflicts=[]
    for (setting,key),rows in sorted(grouped.items()):
        drawers=sorted({e['drawer'] for e in rows})
        item=Counter(e['item'] for e in rows).most_common(1)[0][0]
        record={'setting':setting,'item_key':key,'item':item,'aliases':sorted({e['item'] for e in rows}),
                'head':rows[0]['head'],'drawer':drawers[0] if len(drawers)==1 else None,
                'observed_drawers':drawers,'support_trials':len({e['trial_id'] for e in rows}),
                'support_participants':len({e['participant_id'] for e in rows}),
                'evidence':[{'trial_id':e['trial_id'],'participant_id':e['participant_id'],'query':e['query'],
                             'reply':e['reply'],'source':e['source']} for e in rows],
                'status':'consistent' if len(drawers)==1 else 'conflict'}
        table.append(record)
        if len(drawers)>1:conflicts.append(record)
    return table,conflicts

class Embeddings:
    """BGE semantic embeddings, or the repository's sparse TF-IDF baseline."""
    def __init__(self, backend, texts, sparse_module):
        self.backend=backend;self.sparse=sparse_module;self.cache={}
        if backend=='bge':
            from sentence_transformers import SentenceTransformer
            self.encoder=SentenceTransformer('BAAI/bge-small-en-v1.5',revision='5c38ec7c405ec4b44b94cc5a9bb96e735b38267a',device='cpu')
        elif backend=='tfidf':
            self.idf=sparse_module.fit_idf(tokens(t) for t in texts)
        else:raise ValueError('backend must be bge or tfidf')

    def vectors(self,texts):
        missing=list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            if self.backend=='bge':
                vecs=self.encoder.encode(missing,normalize_embeddings=True,show_progress_bar=False,batch_size=32)
            else:vecs=[self.sparse.embed(tokens(t),self.idf) for t in missing]
            self.cache.update(zip(missing,vecs))
        return [self.cache[t] for t in texts]

    def scores(self,text,others):
        q=self.vectors([text])[0];vs=self.vectors(others)
        return [float(np.dot(q,v)) if self.backend=='bge' else self.sparse.cosine(q,v) for v in vs]