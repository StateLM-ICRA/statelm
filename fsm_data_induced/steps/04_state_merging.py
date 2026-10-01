# @title State merging
def learn_acceptor(frames):
    from aalpy.learning_algs.general_passive.GeneralizedStateMerging import GeneralizedStateMerging
    positive={tuple(tokens(f)) for f in frames}
    negative={p[:i] for p in positive for i in range(len(p))}
    negative.update(tuple(t for t in p if t!='<item>') for p in positive)
    negative-=positive
    data=[(p,True) for p in sorted(positive)]+[(p,False) for p in sorted(negative)]
    machine=GeneralizedStateMerging(output_behavior='moore',transition_behavior='deterministic').run(data,data_format='labeled_sequences')
    prefixes={()}
    for seq,_ in data:prefixes.update(seq[:i] for i in range(1,len(seq)+1))
    return machine,{'algorithm':'AALpy generalized red-blue state merging, RPNI-style',
        'positive_frames':len(positive),'structural_negative_sequences':len(negative),
        'pta_states':len(prefixes),'learned_states':len(machine.states),
        'negative_assumption':'Incomplete prefixes and item-deleted frames reject unless observed as complete positive frames.'}

def traverse(machine,frame):
    state=machine.initial_state;path=[state.state_id]
    for token in tokens(frame):
        if token not in state.transitions:return False,path
        state=state.transitions[token];path.append(state.state_id)
    return bool(state.output),path

@dataclass(frozen=True)
class Thresholds:
    frame:float=.85
    item:float=.82
    margin:float=.05