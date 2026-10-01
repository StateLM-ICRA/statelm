# @title Participant-held-out request evaluation
test_report = evaluate_requests(model, examples['test'])
display(pd.DataFrame([{k:v for k,v in test_report.items() if k!='rows'}]))
display(pd.DataFrame([{'trial':r['trial_id'], 'query':r['query'], 'item_seen_in_training':r['seen_item'],
                       'action':r['prediction']['action'], 'predicted_drawer':r['prediction'].get('drawer'),
                       'source_drawer':r['expected_drawer_from_text'], 'correct_item_and_drawer':r['correct']}
                      for r in test_report['rows']]))
(RESULTS / 'heldout_request_evaluation.json').write_text(json.dumps(test_report, indent=2))
print('This score uses the training-only model. Unseen item locations are not filled from the reference table.')