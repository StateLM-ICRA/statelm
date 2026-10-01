from __future__ import annotations

import copy
import math
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# These words may surround a direct item request without changing its meaning.
FILLER_WORDS = {
    'a', 'an', 'the', 'some', 'please', 'pls', 'i', 'me', 'my', 'want',
    'need', 'would', 'like', 'can', 'could', 'you', 'do', 'have', 'get',
    'give', 'show', 'find', 'locate', 'where', 'is', 'are', 'tell'
}
NEGATION_OR_CORRECTION = {
    'no', 'not', 'dont', 'didnt', 'isnt', 'without', 'except', 'instead', 'wrong'
}
CACHEABLE_ACTIONS = {'locate', 'clarify', 'not_found'}


def words(text):
    return re.findall(r'[a-z0-9]+', str(text).lower())


def latest_user_text(payload):
    for turn in reversed(payload.get('conversation', [])):
        if turn.get('role') == 'user':
            return turn.get('text', '')
    return ''


def state_signature(payload):
    memory = payload.get('memory') or {}
    compact = {
        'decision_mode': payload.get('decision_mode'),
        'state': payload.get('state'),
        'has_pending_family': bool(memory.get('pending_item_family')),
        'known_attribute_keys': sorted((memory.get('known_attributes') or {}).keys()),
        'missing_attributes': sorted(memory.get('missing_attributes') or []),
        'last_robot_action': memory.get('last_robot_action'),
        'has_proposed_answer': payload.get('proposed_robot_answer') is not None,
    }
    return json.dumps(compact, sort_keys=True, separators=(',', ':'))


def _lexicon(inventory):
    entries = {}
    for item in inventory:
        for phrase in {item['name'], *item.get('aliases', [])}:
            token_key = tuple(words(phrase))
            if token_key:
                entries.setdefault(token_key, []).append(('item', item['id'], item))
        family_phrase = item['family'].replace('_', ' ')
        token_key = tuple(words(family_phrase))
        if token_key:
            entries.setdefault(token_key, []).append(('family', item['family'], None))
        for key, value in item.get('attributes', {}).items():
            token_key = tuple(words(value))
            if token_key:
                entries.setdefault(token_key, []).append(('attribute', (key, value), None))
    return entries


def request_features(payload, inventory):
    text = latest_user_text(payload)
    token_list = words(text)
    entries = _lexicon(inventory)
    phrases = sorted(entries, key=lambda value: (-len(value), value))
    priority = {'item': 3, 'family': 2, 'attribute': 1}

    output = []
    slots = {'item_id': None, 'family': None, 'attributes': {}}
    ambiguous = False
    index = 0
    while index < len(token_list):
        matches = []
        for phrase in phrases:
            width = len(phrase)
            if token_list[index:index + width] == list(phrase):
                for descriptor in entries[phrase]:
                    matches.append((width, priority[descriptor[0]], descriptor))
        if not matches:
            if token_list[index] not in FILLER_WORDS:
                output.append(token_list[index])
            index += 1
            continue

        best_width = max(match[0] for match in matches)
        matches = [match for match in matches if match[0] == best_width]
        best_priority = max(match[1] for match in matches)
        descriptors = [match[2] for match in matches if match[1] == best_priority]
        kind = descriptors[0][0]

        if kind == 'item':
            item_ids = {descriptor[1] for descriptor in descriptors}
            if len(item_ids) != 1:
                ambiguous = True
                output.append('<ambiguous-item>')
            else:
                item_id = next(iter(item_ids))
                item = next(item for item in inventory if item['id'] == item_id)
                if slots['item_id'] not in {None, item_id}:
                    ambiguous = True
                slots['item_id'] = item_id
                slots['family'] = item['family']
                slots['attributes'].update(item.get('attributes', {}))
                output.append('<item>')
        elif kind == 'family':
            families = {descriptor[1] for descriptor in descriptors}
            if len(families) != 1:
                ambiguous = True
                output.append('<ambiguous-family>')
            else:
                family = next(iter(families))
                if slots['family'] not in {None, family}:
                    ambiguous = True
                slots['family'] = family
                output.append('<family>')
        else:
            values = {descriptor[1] for descriptor in descriptors}
            if len(values) != 1:
                ambiguous = True
                output.append('<ambiguous-attribute>')
            else:
                key, value = next(iter(values))
                previous = slots['attributes'].get(key)
                if previous not in {None, value}:
                    ambiguous = True
                slots['attributes'][key] = value
                output.append(f'<attr:{key}>')
        index += best_width

    surface = ' '.join(output) or '<empty>'
    return {
        'text': text,
        'raw_normalized': ' '.join(token_list),
        'surface': surface,
        'slots': slots,
        'ambiguous': ambiguous,
        'has_negation_or_correction': bool(set(token_list) & NEGATION_OR_CORRECTION),
        'variant': json.dumps(slots, sort_keys=True, separators=(',', ':')),
    }


def surface_similarity(left, right):
    left_tokens, right_tokens = left.split(), right.split()
    left_set, right_set = set(left_tokens), set(right_tokens)
    union = left_set | right_set
    jaccard = len(left_set & right_set) / len(union) if union else 1.0
    sequence = SequenceMatcher(None, left, right).ratio()
    return 0.55 * sequence + 0.45 * jaccard


def decisions_equal(left, right):
    if left is None or right is None:
        return False
    left_dict = left.model_dump() if hasattr(left, 'model_dump') else dict(left)
    right_dict = right.model_dump() if hasattr(right, 'model_dump') else dict(right)
    return left_dict == right_dict


def make_template(decision, payload, features, inventory):
    data = decision.model_dump() if hasattr(decision, 'model_dump') else dict(decision)
    if payload.get('decision_mode') != 'dialogue_decision':
        return None
    if data['action'] not in CACHEABLE_ACTIONS or data['failure_detected']:
        return None
    if data['action'] == 'clarify' and data['item_family'] is None:
        return None

    by_id = {item['id']: item for item in inventory}
    item_slot = by_id.get(features['slots']['item_id'])
    memory = payload.get('memory') or {}

    if item_slot and item_slot['family'] == data['item_family']:
        family_source = 'input_item'
        fixed_family = None
    elif features['slots']['family'] == data['item_family']:
        family_source = 'input_family'
        fixed_family = None
    elif memory.get('pending_item_family') == data['item_family']:
        family_source = 'pending_family'
        fixed_family = None
    elif data['item_family'] is None:
        family_source = 'none'
        fixed_family = None
    else:
        family_source = 'fixed'
        fixed_family = data['item_family']

    attribute_sources = {}
    for key, value in data['attributes'].items():
        if features['slots']['attributes'].get(key) == value:
            attribute_sources[key] = {'source': 'input_attribute', 'value': None}
        elif item_slot and item_slot.get('attributes', {}).get(key) == value:
            attribute_sources[key] = {'source': 'input_item', 'value': None}
        elif (memory.get('known_attributes') or {}).get(key) == value:
            attribute_sources[key] = {'source': 'memory', 'value': None}
        else:
            attribute_sources[key] = {'source': 'fixed', 'value': value}

    if data['resolved_item_id'] is None:
        item_source = 'none'
    elif features['slots']['item_id'] == data['resolved_item_id']:
        item_source = 'input_item'
    else:
        target_item = by_id.get(data['resolved_item_id'])
        if not target_item:
            return None
        matches = [item for item in inventory if item['family'] == data['item_family'] and
                   all(item.get('attributes', {}).get(key) == value
                       for key, value in data['attributes'].items())]
        if len(matches) != 1:
            return None
        item_source = 'resolve_unique'

    return {
        'action': data['action'],
        'family_source': family_source,
        'fixed_family': fixed_family,
        'item_source': item_source,
        'attribute_sources': attribute_sources,
        'missing_attributes': list(data['missing_attributes']),
        'failure_detected': False,
        'failure_type': 'none',
        'recovery_strategy': 'none',
    }


def template_signature(template):
    return json.dumps(template, sort_keys=True, separators=(',', ':'))


def instantiate_template(template, payload, features, inventory):
    if features['ambiguous'] or features['has_negation_or_correction']:
        return None
    by_id = {item['id']: item for item in inventory}
    input_item = by_id.get(features['slots']['item_id'])
    memory = payload.get('memory') or {}

    source = template['family_source']
    if source == 'input_item':
        family = input_item['family'] if input_item else None
    elif source == 'input_family':
        family = features['slots']['family']
    elif source == 'pending_family':
        family = memory.get('pending_item_family')
    elif source == 'fixed':
        family = template['fixed_family']
    else:
        family = None

    attributes = {}
    for key, specification in template['attribute_sources'].items():
        attr_source = specification['source']
        if attr_source == 'input_attribute':
            value = features['slots']['attributes'].get(key)
        elif attr_source == 'input_item':
            value = input_item.get('attributes', {}).get(key) if input_item else None
        elif attr_source == 'memory':
            value = (memory.get('known_attributes') or {}).get(key)
        else:
            value = specification['value']
        if value is None:
            return None
        attributes[key] = value

    action = template['action']
    resolved_item_id = None
    if action == 'locate':
        if template['item_source'] == 'input_item':
            item = input_item
        else:
            matches = [item for item in inventory if item['family'] == family and
                       all(item.get('attributes', {}).get(key) == value
                           for key, value in attributes.items())]
            item = matches[0] if len(matches) == 1 else None
        if item is None:
            return None
        family = item['family']
        resolved_item_id = item['id']
        attributes = dict(item.get('attributes', {}))
    elif action == 'clarify':
        matching_family = [item for item in inventory if item['family'] == family and
                           all(item.get('attributes', {}).get(key) == value
                               for key, value in attributes.items())]
        valid_keys = {key for item in matching_family for key in item.get('attributes', {})}
        if len(matching_family) < 2 or not set(template['missing_attributes']) <= valid_keys:
            return None
    elif action == 'not_found':
        matches = [item for item in inventory if item['family'] == family and
                   all(item.get('attributes', {}).get(key) == value
                       for key, value in attributes.items())]
        if matches:
            return None

    decision = Runtime.Decision(
        action=action,
        item_family=family,
        resolved_item_id=resolved_item_id,
        attributes=attributes,
        missing_attributes=list(template['missing_attributes']),
        failure_detected=template['failure_detected'],
        failure_type=template['failure_type'],
        recovery_strategy=template['recovery_strategy'],
    )
    return Runtime.validate_decision(decision, payload, inventory)


def exact_seed_rule(payload, features, inventory):
    # The only initial direct rule: an exact, non-negated inventory item request.
    if payload.get('decision_mode') != 'dialogue_decision':
        return None
    if payload.get('state') != 'awaiting_request':
        return None
    if features['ambiguous'] or features['has_negation_or_correction']:
        return None
    if features['surface'] != '<item>' or not features['slots']['item_id']:
        return None
    item = next((item for item in inventory if item['id'] == features['slots']['item_id']), None)
    if item is None:
        return None
    decision = Runtime.Decision(
        action='locate', item_family=item['family'], resolved_item_id=item['id'],
        attributes=dict(item.get('attributes', {})), missing_attributes=[],
        failure_detected=False, failure_type='none', recovery_strategy='none'
    )
    return Runtime.validate_decision(decision, payload, inventory)


@dataclass
class CandidatePattern:
    pattern_id: str
    state_key: str
    template: dict
    surfaces: set[str] = field(default_factory=set)
    examples: set[str] = field(default_factory=set)
    variants: set[str] = field(default_factory=set)
    confirmed: int = 0
    shadow_successes: int = 0
    shadow_failures: int = 0
    served_successes: int = 0
    served_failures: int = 0
    heat: float = 0.0
    last_step: int = 0
    status: str = 'candidate'

    def decay_to(self, step):
        elapsed = max(0, step - self.last_step)
        self.heat *= HEAT_DECAY ** elapsed
        self.last_step = step

    @property
    def shadow_trials(self):
        return self.shadow_successes + self.shadow_failures

    @property
    def shadow_accuracy(self):
        return self.shadow_successes / self.shadow_trials if self.shadow_trials else 0.0


class PatternCache:
    def __init__(self):
        self.patterns = []
        self.next_id = 1
        self.promotion_events = []

    def decay_all(self, step):
        for pattern in self.patterns:
            pattern.decay_to(step)

    def score(self, pattern, state_key, surface):
        if pattern.state_key != state_key:
            return 0.0
        return max((surface_similarity(surface, example)
                    for example in pattern.surfaces), default=0.0)

    def learn_confirmed(self, payload, inventory, decision, step):
        features = request_features(payload, inventory)
        template = make_template(decision, payload, features, inventory)
        if template is None or features['ambiguous'] or features['has_negation_or_correction']:
            return None

        state_key = state_signature(payload)
        signature = template_signature(template)
        compatible = [pattern for pattern in self.patterns
                      if pattern.state_key == state_key and
                      template_signature(pattern.template) == signature]
        scored = [(self.score(pattern, state_key, features['surface']), pattern)
                  for pattern in compatible]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        if scored and scored[0][0] >= CACHE_MERGE_THRESHOLD:
            pattern = scored[0][1]
        else:
            pattern = CandidatePattern(
                pattern_id=f'pattern_{self.next_id:04d}',
                state_key=state_key,
                template=template,
                last_step=step,
            )
            self.next_id += 1
            self.patterns.append(pattern)

        pattern.decay_to(step)
        pattern.surfaces.add(features['surface'])
        pattern.examples.add(features['raw_normalized'])
        pattern.variants.add(features['variant'])
        pattern.confirmed += 1
        pattern.heat += 1.0
        self.maybe_promote(pattern, step)
        return pattern.pattern_id

    def shadow_test(self, payload, inventory, target, step):
        features = request_features(payload, inventory)
        state_key = state_signature(payload)
        for pattern in self.patterns:
            if pattern.status != 'candidate':
                continue
            score = self.score(pattern, state_key, features['surface'])
            if score < FSM_USE_THRESHOLD:
                continue
            try:
                proposed = instantiate_template(pattern.template, payload, features, inventory)
            except Exception:
                proposed = None
            if proposed is None:
                continue
            if decisions_equal(proposed, target):
                pattern.shadow_successes += 1
            else:
                pattern.shadow_failures += 1
            self.maybe_promote(pattern, step)

    def maybe_promote(self, pattern, step):
        pattern.decay_to(step)
        ready = (
            pattern.status == 'candidate' and
            pattern.confirmed >= PROMOTION_MIN_CONFIRMED and
            pattern.shadow_trials >= PROMOTION_MIN_SHADOW and
            len(pattern.examples) >= PROMOTION_MIN_EXAMPLES and
            len(pattern.variants) >= PROMOTION_MIN_VARIANTS and
            pattern.shadow_accuracy >= PROMOTION_MIN_ACCURACY and
            pattern.heat >= PROMOTION_MIN_HEAT
        )
        if ready:
            pattern.status = 'promoted'
            self.promotion_events.append({'step': step, 'pattern_id': pattern.pattern_id})

    def demote(self, pattern, step):
        pattern.status = 'candidate'
        pattern.served_failures += 1
        pattern.heat *= 0.5
        pattern.last_step = step


class AdaptiveRouter:
    def __init__(self, cache):
        self.cache = cache

    def try_fsm(self, payload, inventory, step):
        self.cache.decay_all(step)
        features = request_features(payload, inventory)
        try:
            seed = exact_seed_rule(payload, features, inventory)
        except Exception:
            seed = None
        if seed is not None:
            return seed, {'source': 'fsm_seed_exact_item', 'pattern_id': None, 'match_score': 1.0}

        if features['ambiguous'] or features['has_negation_or_correction']:
            return None, {'source': 'slm', 'reason': 'ambiguous_or_corrective_language'}

        state_key = state_signature(payload)
        scored = []
        for pattern in self.cache.patterns:
            if pattern.status != 'promoted':
                continue
            score = self.cache.score(pattern, state_key, features['surface'])
            if score > 0:
                scored.append((score, pattern))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        if not scored or scored[0][0] < FSM_USE_THRESHOLD:
            return None, {'source': 'slm', 'reason': 'no_promoted_match'}
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if scored[0][0] - second_score < MIN_TOP2_MARGIN:
            return None, {'source': 'slm', 'reason': 'competing_patterns'}

        score, pattern = scored[0]
        try:
            decision = instantiate_template(pattern.template, payload, features, inventory)
        except Exception:
            decision = None
        if decision is None:
            return None, {'source': 'slm', 'reason': 'template_binding_failed'}
        return decision, {
            'source': 'fsm_promoted',
            'pattern_id': pattern.pattern_id,
            'match_score': score,
        }


print('Adaptive FSM/cache definitions loaded.')

