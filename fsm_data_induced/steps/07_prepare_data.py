# @title Select embeddings and prepare data
BACKEND = "bge" # @param ["bge", "tfidf"]
import spacy, importlib.util
nlp = spacy.load('en_core_web_sm')
extractor = PhraseExtractor(nlp)
spec = importlib.util.spec_from_file_location('repo_similarity', DATA / 'repo_similarity.py')
sparse_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(sparse_module)
corpus = json.loads((DATA / 'success_corpus.json').read_text())
split_data, participant_groups = split_trials(corpus)
examples, extraction_audits = {}, {}
for split, trials in split_data.items():
    examples[split], extraction_audits[split] = extract_examples(trials, extractor)
display(pd.DataFrame([{'split': s, 'participants': len(participant_groups[s]),
                       'success_trials': len(split_data[s]), 'accepted_exchanges': len(examples[s]),
                       'excluded_trials': len(split_data[s])-len(examples[s])} for s in split_data]))
display(pd.DataFrame(examples['train'])[['trial_id','setting','query','item','reply','drawer','frame']])
(RESULTS / 'split_manifest.json').write_text(json.dumps(participant_groups, indent=2))
(RESULTS / 'extraction_audit_by_split.json').write_text(json.dumps(extraction_audits, indent=2))