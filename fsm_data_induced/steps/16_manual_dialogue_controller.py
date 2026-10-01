# @title Explicit dialogue FSM: clarification and items not found
# This controller is manually specified. It uses only the learned inventory;
# it never reads the reference table or invents new item–drawer pairs.
# "OBJECT_MISSING" below means absent from the LEARNED inventory, not proof
# that the item does not exist in the physical cart.

class ClarificationMissingFSM:
    QUERY = 'WAITING_FOR_QUERY'
    CLARIFY = 'WAITING_FOR_CLARIFICATION'
    RULES = {
        (QUERY, 'ONE_MATCH'): ('RETURN_LOCATION', QUERY),
        (QUERY, 'MULTIPLE_MATCHES'): ('ASK_WHICH_ONE', CLARIFY),
        (QUERY, 'NO_LEARNED_MATCH'): ('OBJECT_MISSING', QUERY),
        (QUERY, 'UNSUPPORTED'): ('UNSUPPORTED_REQUEST', QUERY),
        (QUERY, 'LOCATION_CONFLICT'): ('LOCATION_CONFLICT', QUERY),
        (QUERY, 'SETTING_REQUIRED'): ('SETTING_REQUIRED', QUERY),
        (CLARIFY, 'ONE_MATCH'): ('RETURN_LOCATION', QUERY),
        (CLARIFY, 'MULTIPLE_MATCHES'): ('ASK_WHICH_ONE', CLARIFY),
        (CLARIFY, 'NO_CANDIDATE_MATCH'): ('ASK_WHICH_ONE', CLARIFY),
        (CLARIFY, 'LOCATION_CONFLICT'): ('LOCATION_CONFLICT', QUERY),
        (CLARIFY, 'NEW_REQUEST'): ('RESTART_REQUEST', QUERY),
    }

    def __init__(self, model, setting='lab'):
        self.model, self.setting = model, setting
        self.state, self.pending, self.history = self.QUERY, [], []

    def _step(self, event, result):
        before = self.state
        action, after = self.RULES[(before, event)]
        self.state = after
        result = {**result, 'action': action, 'event': event,
                  'state_before': before, 'state_after': after,
                  'controller': 'manually_specified_dialogue_fsm'}
        if after == self.QUERY:
            self.pending = []
        self.history.append(result)
        return result

    def _question(self, prefix=''):
        choices = '; '.join(f'{i+1}. {r["item"]}' for i, r in enumerate(self.pending))
        return {'text': prefix + 'Which item do you mean? ' + choices,
                'options': [r['item'] for r in self.pending], 'drawer': None}

    def ask(self, text):
        text = text.strip()
        if self.state == self.CLARIFY:
            # A new location request replaces the old clarification context.
            if re.search(r'\b(where|find|locate|need|want|get|give|show|look for)\b', text, re.I):
                self._step('NEW_REQUEST', {'text': '', 'drawer': None})
            else:
                if re.fullmatch(r'\d+', text):
                    index = int(text)-1
                    matches = self.pending[index:index+1] if 0 <= index < len(self.pending) else []
                else:
                    words = set(tokens(self.model.extractor.key(text))) - {'the','one','please','it','be'}
                    matches = [r for r in self.pending
                               if words and words <= set(tokens(r['item_key']))]
                if len(matches) == 1:
                    result = self.model.render(matches[0])
                    event = 'ONE_MATCH' if result['action']=='RETURN_LOCATION' else 'LOCATION_CONFLICT'
                    return self._step(event, result)
                if matches:
                    self.pending = matches
                    return self._step('MULTIPLE_MATCHES', self._question())
                return self._step('NO_CANDIDATE_MATCH', self._question('That does not match the listed choices. '))

        result = self.model.request(text, self.setting)
        event = {
            'RETURN_LOCATION': 'ONE_MATCH', 'ASK_WHICH_ONE': 'MULTIPLE_MATCHES',
            'ITEM_NOT_LEARNED': 'NO_LEARNED_MATCH', 'UNSUPPORTED_REQUEST': 'UNSUPPORTED',
            'LOCATION_CONFLICT': 'LOCATION_CONFLICT', 'SETTING_REQUIRED': 'SETTING_REQUIRED',
        }[result['action']]
        if event == 'MULTIPLE_MATCHES':
            self.pending = [r for r in self.model.table
                            if r['setting']==self.setting and r['item_key'] in result['options']]
            result = {**result, **self._question()}
        elif event == 'NO_LEARNED_MATCH':
            result = {**result, 'text': "I couldn't find that item in the learned inventory. No location is available.",
                      'drawer': None, 'missing_scope': 'learned_inventory_only'}
        return self._step(event, result)


# Activate this controller for the interactive tester in the next cell.
# The induced recognizer, learned pairs, and successful response template stay in full_model.
Session = ClarificationMissingFSM
if 'live' in globals():
    live['session'] = Session(full_model, scope.value)

display(Markdown('**Rule-based dialogue FSM.** Ambiguous requests ask for a choice; unknown items return no drawer.'))
rule_rows = [{'state': s, 'event': event, 'action': action, 'next_state': nxt}
             for (s,event),(action,nxt) in ClarificationMissingFSM.RULES.items()]
display(pd.DataFrame(rule_rows))

# Synthetic test utterances exercise the controller; they are NOT training facts.
synthetic_dialogues = {
    'clarification': ['Where is tape?', 'Blue.'],
    'invalid clarification, then correction': ['Where is tape?', 'Purple.', 'White.'],
    'item not found': ['Where is the pen?'],
    'new missing request clears old choices': ['Where is tape?', 'Where is my wallet?'],
}
synthetic_rows = []
for scenario, utterances in synthetic_dialogues.items():
    session = Session(full_model, 'lab')
    for utterance in utterances:
        result = session.ask(utterance)
        synthetic_rows.append({'scenario': scenario, 'user': utterance, **result})
display(pd.DataFrame(synthetic_rows)[['scenario','user','text','action','drawer','state_before','state_after']])

checks_session = Session(full_model, 'lab')
assert checks_session.ask('Where is tape?')['action'] == 'ASK_WHICH_ONE'
assert checks_session.state == checks_session.CLARIFY
assert checks_session.ask('Purple.')['event'] == 'NO_CANDIDATE_MATCH'
assert checks_session.state == checks_session.CLARIFY
chosen = checks_session.ask('Blue.')
assert chosen['action'] == 'RETURN_LOCATION' and chosen['item_key'] == 'blue tape'
assert checks_session.state == checks_session.QUERY and not checks_session.pending
missing = checks_session.ask('Where is the pen?')
assert missing['action'] == 'OBJECT_MISSING' and missing['drawer'] is None
checks_session.ask('Where is tape?')
missing = checks_session.ask('Where is my wallet?')
assert missing['action'] == 'OBJECT_MISSING' and missing['drawer'] is None
assert checks_session.state == checks_session.QUERY and not checks_session.pending
print('Clarification, invalid choices, missing items, and context-reset checks passed.')

# Export the explicit controller separately from the data-induced request graph.
from graphviz import Digraph
dialogue_graph = Digraph('manual_dialogue_fsm')
dialogue_graph.attr(rankdir='LR', label='Manual clarification and missing-item controller', labelloc='t')
for state in [ClarificationMissingFSM.QUERY, ClarificationMissingFSM.CLARIFY]:
    dialogue_graph.node(state, label=state.replace('_','\n'), shape='box')
grouped_edges = defaultdict(list)
for row in rule_rows:
    grouped_edges[row['state'],row['next_state']].append(row['event']+' / '+row['action'])
for (source,target),labels in grouped_edges.items():
    dialogue_graph.edge(source,target,label='\n'.join(labels),style='dashed')
(RESULTS / 'manual_dialogue_fsm.dot').write_text(dialogue_graph.source)
(RESULTS / 'manual_dialogue_rules.json').write_text(json.dumps(rule_rows,indent=2))
(RESULTS / 'synthetic_dialogue_checks.json').write_text(json.dumps(synthetic_rows,indent=2))
if shutil.which('dot'):
    controller_svg = dialogue_graph.pipe(format='svg')
    (RESULTS / 'manual_dialogue_fsm.svg').write_bytes(controller_svg)
    display(SVG(controller_svg))
