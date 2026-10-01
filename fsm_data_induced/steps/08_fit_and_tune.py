# @title Fit, select a threshold using validation, and freeze the model
model = InducedLocator(examples['train'], extractor, BACKEND, sparse_module)
tuning = tune_on_validation(model, examples['validation'])
print('Automatically selected response template:', model.template)
display(pd.DataFrame([model.graph_stats]))
print('Validation threshold selection:', tuning)
(RESULTS / 'training_model.json').write_text(json.dumps(model.artifact(), indent=2))
(RESULTS / 'validation_selection.json').write_text(json.dumps(tuning, indent=2))