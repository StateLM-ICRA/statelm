# @title Build the all-source reconstruction
all_examples, all_audit = extract_examples(corpus, extractor)
full_model = InducedLocator(all_examples, extractor, BACKEND, sparse_module)
full_model.thresholds = model.thresholds
print(f'{len(all_examples)} supported exchanges / {len(corpus)} success trials; {len(full_model.table)} learned entries')
print('Learned response wording:', full_model.template)
learned_table = pd.DataFrame([{k:v for k,v in r.items() if k!='evidence'} for r in full_model.table])
display(learned_table[['setting','item','drawer','support_trials','support_participants','status']])
display(pd.DataFrame([{'trial':r['trial_id'],'reason':r.get('reason')} for r in all_audit if r['status']=='excluded']))
(RESULTS / 'all_source_learned_model.json').write_text(json.dumps(full_model.artifact(), indent=2))
(RESULTS / 'all_source_evidence.json').write_text(json.dumps(all_examples, indent=2))
(RESULTS / 'all_source_extraction_audit.json').write_text(json.dumps(all_audit, indent=2))
learned_table.to_csv(RESULTS / 'learned_item_drawer_pairs.csv', index=False)