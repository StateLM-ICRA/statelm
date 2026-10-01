# @title Compare the frozen learned table with the reference
frozen_before = fingerprint(full_model)
reference = json.loads((DATA / 'evaluation_inventory.json').read_text())
inventory_report = compare_inventory(full_model.table, reference, extractor)
assert fingerprint(full_model) == frozen_before, 'Evaluation must not change the model'
display(pd.DataFrame([{k:v for k,v in inventory_report.items() if k not in ['rows','missing_reference_items','interpretation']}]))
comparison_table = pd.DataFrame(inventory_report['rows'])
display(comparison_table)
review_queue = comparison_table[comparison_table['status']!='correct']
display(Markdown('**Pairs requiring review**'))
display(review_queue)
display(Markdown('**Reference items not recovered from the success data**'))
display(pd.DataFrame(inventory_report['missing_reference_items']))
comparison_table.to_csv(RESULTS / 'ground_truth_comparison.csv', index=False)
review_queue.to_csv(RESULTS / 'review_queue.csv', index=False)
pd.DataFrame(inventory_report['missing_reference_items']).to_csv(RESULTS / 'missing_reference_items.csv', index=False)
(RESULTS / 'inventory_evaluation.json').write_text(json.dumps(inventory_report, indent=2))