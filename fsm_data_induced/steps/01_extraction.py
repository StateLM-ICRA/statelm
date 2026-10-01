# @title Text and noun-phrase extraction
"""Success-dialogue induction. This module never reads a reference inventory."""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from functools import lru_cache
import hashlib, json, re
from pathlib import Path
import numpy as np

NUMBERS = dict(zip('zero one two three four five six seven eight nine ten'.split(), range(11)))
NUMBER_RE = r'(?:\d+|' + '|'.join(NUMBERS) + r')'
DRAWER_RE = re.compile(r'\bdrawer\s+(' + NUMBER_RE + r')\b', re.I)
GENERIC_NOUNS = {'drawer','item','object','request','robot','cart','trial','interaction','part','study','one','something','anything','thing','number','end','time','pair'}

def tokens(text):
    return re.findall(r'<item>|[^\W_]+(?:[\x27’-][^\W_]+)*', text.casefold())

def normalized(text):
    return ' '.join(tokens(text))

def source_words(text):
    text=re.sub(r'\[[^\]]*s\]', ' ', text)
    text=re.sub(r'\b(?:door|draw)\s+(?='+NUMBER_RE+r'\b)', 'drawer ', text, flags=re.I)
    text=re.sub(r'\bdrawer\s+(?:too|to)\b', 'drawer two', text, flags=re.I)
    return [str(NUMBERS.get(t,t)) for t in tokens(text)]

def drawer_statement(text):
    # General domain assumption: an explicit drawer number denotes the location.
    matches=list(DRAWER_RE.finditer(text))
    if len(matches)!=1 or re.search(r"\?|\b(?:not|never|isn't|maybe|perhaps|might|think|instead)\b", text, re.I):
        return None
    m=matches[0]; value=NUMBERS.get(m[1].casefold())
    if value is None: value=int(m[1])
    if value<=0:return None
    template=text[:m.start(1)]+'{drawer}'+text[m.end(1):]
    return value, re.sub(r'\s+',' ',template).strip()

class PhraseExtractor:
    def __init__(self, nlp): self.nlp=nlp

    @lru_cache(maxsize=8192)
    def key(self, text):
        doc=self.nlp(text)
        return ' '.join(t.lemma_.casefold() for t in doc if not t.is_punct and t.pos_!='DET').strip()

    @lru_cache(maxsize=8192)
    def candidates(self, text):
        doc=self.nlp(text); result=[]
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ not in {'NOUN','PROPN'}:continue
            if chunk.root.lemma_.casefold() in GENERIC_NOUNS:continue
            kept=[t for t in chunk if t.pos_ not in {'DET','PRON'} and not t.is_punct]
            if not kept:continue
            start,end=kept[0].idx,kept[-1].idx+len(kept[-1])
            surface=text[start:end]; key=self.key(surface)
            if key and len(tokens(surface))<=10:
                result.append({'text':surface,'key':key,'start':start,'end':end,
                               'head':chunk.root.lemma_.casefold(),'dependency':chunk.root.dep_})
        return tuple(result)

    def choose(self, text, operator_context=''):
        candidates=list({c['key']:c for c in self.candidates(text)}.values())
        if not candidates:return None,'no_noun_phrase'
        context=self.key(operator_context)
        def score(c):
            repeated=bool(c['key'] and c['key'] in context)
            return 3*repeated + .25*len(tokens(c['text'])) + .4*(c['dependency']!='ROOT')
        ranked=sorted(candidates,key=lambda c:(score(c),c['start']),reverse=True)
        if len(ranked)>1 and abs(score(ranked[0])-score(ranked[1]))<.2:
            return None,'ambiguous_noun_phrases'
        return ranked[0],None

    def frame(self, text, phrase):
        # Keep the sentence containing the selected mention, excluding trailing study chatter.
        for sentence in self.nlp(text).sents:
            if sentence.start_char<=phrase['start']<sentence.end_char:
                text=sentence.text
                phrase={**phrase,'start':phrase['start']-sentence.start_char,'end':phrase['end']-sentence.start_char}
                break
        # Mask every repeated lemma-equivalent mention in the selected sentence.
        spans=[(c['start'],c['end']) for c in self.candidates(text) if c['key']==phrase['key']]
        if not spans:spans=[(phrase['start'],phrase['end'])]
        for start,end in sorted(spans,reverse=True):text=text[:start]+'<item>'+text[end:]
        return normalized(text)