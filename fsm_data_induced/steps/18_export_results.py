# @title Export the run (optional browser download)
import importlib.metadata
summary={'accepted_success_exchanges':len(all_examples),'total_success_trials':len(corpus),
 'learned_entries':len(full_model.table),'learned_template':full_model.template,
 'graph':full_model.graph_stats,'backend':BACKEND,
 'reference_comparison':{k:v for k,v in inventory_report.items() if k not in ['rows','missing_reference_items']},
 'heldout_requests':{k:v for k,v in test_report.items() if k!='rows'},
 'scope':'Exploratory participant-held-out request test plus separate all-source reconstruction.'}
(RESULTS / 'summary.json').write_text(json.dumps(summary,indent=2))
(RESULTS / 'source_manifest.json').write_text(json.dumps(source_manifest,indent=2))
(RESULTS / 'environment.json').write_text(json.dumps({'python':sys.version,
 'packages':{p:importlib.metadata.version(p) for p in ['aalpy','spacy','en-core-web-sm','sentence-transformers','transformers','numpy','pandas']},
 'bge_model':'BAAI/bge-small-en-v1.5','bge_revision':'5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'},indent=2))
archive_path=BASE / 'success_fsm_results.zip'
with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(RESULTS.rglob('*')):
        if path.is_file(): archive.write(path,path.relative_to(RESULTS))
print('Saved success_fsm_results.zip with',len(list(RESULTS.iterdir())),'result files.')
DOWNLOAD_ZIP = False # @param {type:"boolean"}
if DOWNLOAD_ZIP and IN_COLAB:
    from google.colab import files
    files.download(str(archive_path))
display(pd.DataFrame([{'quantity':'learned pairs','value':len(full_model.table)},
 {'quantity':'correct mapped pairs','value':inventory_report['status_counts'].get('correct',0)},
 {'quantity':'reference items','value':inventory_report['reference_entries']},
 {'quantity':'held-out correct / extracted test cases','value':str(test_report['correct'])+' / '+str(test_report['cases'])}]))