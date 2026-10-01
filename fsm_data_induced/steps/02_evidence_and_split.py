# @title Evidence extraction and participant split
def extract_examples(trials, extractor):
    examples=[]; audit=[]
    for trial in trials:
        attempts=[]; options=[]
        if trial['label']!='success':raise ValueError('Success-only extractor received another label')
        if trial['setting'] not in {'lab','hospital'}:
            audit.append({'trial_id':trial['trial_id'],'status':'excluded','reason':'unresolved_supplied_setting'});continue
        for candidate in trial['candidates']:
            history=[]
            for i,turn in enumerate(candidate['turns']):
                statement=drawer_statement(turn['text']) if turn['role']=='robot' else None
                if statement:
                    participant=[t for t in history if t['role']=='participant']
                    operator=' '.join(t['text'] for t in history if t['role']=='operator')
                    if not participant:continue
                    request=participant[-1]['text']; phrase,reason=extractor.choose(request,operator)
                    base={'trial_id':trial['trial_id'],'source':candidate['source'],'robot_turn':i}
                    if phrase is None:
                        attempts.append({**base,'reason':reason});continue
                    drawer,template=statement
                    best_support=0.; item_support=0.; drawer_supported=False
                    for raw in trial['raw_transcripts']:
                        raw_words=source_words(raw); reply_words=source_words(turn['text'])
                        blocks=SequenceMatcher(None,reply_words,raw_words,autojunk=False).get_matching_blocks()
                        coverage=sum(b.size for b in blocks)/max(len(reply_words),1)
                        source_string=' '.join(raw_words)
                        has_drawer=bool(re.search(r'\bdrawer '+str(drawer)+r'\b',source_string))
                        mention_words=source_words(phrase['text'])
                        item_fraction=sum(w in raw_words for w in mention_words)/max(len(mention_words),1)
                        if has_drawer and coverage>best_support:
                            best_support=coverage;item_support=item_fraction;drawer_supported=True
                    if not drawer_supported or best_support<.7 or item_support<.5:
                        attempts.append({**base,'reason':'insufficient_raw_text_support','reply_support':best_support,'item_support':item_support});continue
                    frame=extractor.frame(request,phrase)
                    example={**base,'participant_id':trial['participant_id'],'setting':trial['setting'],
                         'query':request,'reply':turn['text'],'item':phrase['text'],'item_key':phrase['key'],
                         'head':phrase['head'],'drawer':drawer,'template':template,'frame':frame,
                         'reply_raw_support':best_support,'item_raw_support':item_support,
                         'role_source_kind':candidate['kind']}
                    # Prefer the original provided archive; then strongest source support.
                    rank=(candidate['kind']=='provided_role_archive',best_support,item_support,-i)
                    options.append((rank,example))
                    break
                history.append(turn)
        if options:
            chosen=max(options,key=lambda x:x[0])[1];examples.append(chosen)
            audit.append({'trial_id':trial['trial_id'],'status':'accepted','item':chosen['item'],'drawer':chosen['drawer'],
                          'source':chosen['source'],'attempts':attempts})
        else:audit.append({'trial_id':trial['trial_id'],'status':'excluded','reason':'no_supported_user_robot_pair','attempts':attempts})
    return examples,audit

def split_trials(trials):
    # Hashing participant IDs is independent of utterances, extraction results and gold.
    participants=sorted({t['participant_id'] for t in trials},key=lambda p:hashlib.sha256(('fsm-v2:'+p).encode()).hexdigest())
    n=len(participants);a=max(1,int(.65*n));b=max(a+1,int(.82*n))
    groups={'train':participants[:a],'validation':participants[a:b],'test':participants[b:]}
    assignment={p:s for s,ps in groups.items() for p in ps}
    return {s:[t for t in trials if assignment[t['participant_id']]==s] for s in groups},groups