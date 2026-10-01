# @title Integrity checks: evidence dependence, conflicts, no reference-table feedback
import copy
checks = []
for a,b in [('train','validation'),('train','test'),('validation','test')]:
    assert not set(participant_groups[a]) & set(participant_groups[b])
checks.append('Participants are disjoint across splits')
assert set(model.artifact()['training_trials']) <= {e['trial_id'] for e in examples['train']}
checks.append('Training model uses only training trials')
for frame in full_model.frames: assert traverse(full_model.machine,frame)[0]
checks.append('Learned graph accepts every positive training prototype')
assert full_model.template in [e['template'] for e in all_examples]
checks.append('Successful response wording occurs in the supplied replies')
assert drawer_statement('It is not in drawer two.') is None
assert drawer_statement('Drawer two or drawer three.') is None
checks.append('Negated and multiple-drawer replies are excluded')
fixture = {'trial_id':'fixture', 'participant_id':'fixture', 'label':'success', 'setting':'lab',
 'raw_transcripts':['Where is the copper widget? Item is in drawer seven.'],
 'candidates':[{'source':'fixture','kind':'provided_role_archive','turns':[
 {'role':'participant','text':'Where is the copper widget?'},
 {'role':'robot','text':'Item is in drawer seven.'}]}]}
ex1,_ = extract_examples([fixture],extractor)
fixture['raw_transcripts'][0] = fixture['raw_transcripts'][0].replace('seven','nine')
fixture['candidates'][0]['turns'][1]['text'] = 'Item is in drawer nine.'
ex2,_ = extract_examples([fixture],extractor)
assert ex1[0]['drawer']==7 and ex2[0]['drawer']==9
checks.append('Changing an observed reply changes the extracted drawer')
conflicting = copy.deepcopy(all_examples[0]); conflicting['drawer']=987; conflicting['trial_id']='conflict_fixture'
_, conflicts = build_table(all_examples+[conflicting])
assert any(c['item_key']==conflicting['item_key'] and c['drawer'] is None for c in conflicts)
checks.append('Contradictory locations remain withheld')
before = fingerprint(full_model)
corrupted_reference = copy.deepcopy(reference)
for r in corrupted_reference['items']: r['drawer']=999
compare_inventory(full_model.table,corrupted_reference,extractor)
assert fingerprint(full_model)==before
checks.append('Changing evaluation ground truth cannot alter the model')
for question in ['Where is my wallet?','Where is the red tape?','Tell me a joke.','Play music.']:
    assert full_model.request(question,'lab').get('drawer') is None
checks.append('Unknown items / unrelated demo requests do not produce a drawer')
assert full_model.request('Where is SD card?','hospital').get('drawer') is None
checks.append('Facts do not cross settings')
test_session=Session(full_model,'lab')
assert test_session.ask('Where is tape?')['action']=='ASK_WHICH_ONE'
assert test_session.ask('Blue.')['item_key']=='blue tape'
test_session.ask('Where is tape?'); test_session.ask('Where is my wallet?')
assert not test_session.pending
checks.append('Clarification narrows candidates and a new request clears them')
display(pd.DataFrame({'passed':checks}))
(RESULTS / 'integrity_checks.json').write_text(json.dumps({'passed':checks},indent=2))