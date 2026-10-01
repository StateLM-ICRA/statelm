"""Two isolated simulations using the existing saved-model inference code."""
import argparse
import copy
from contextlib import redirect_stdout
import json
import math
from pathlib import Path
import re
import sys
import traceback
from statelm.playground import Session, load_session

ERRORS = (
    ('wrong_drawer', 'Wrong drawer'),
    ('wrong_item', 'Wrong item'),
    ('paper_face_mask', 'Paper request → face-mask answer'),
    ('vague', 'Vague drawer instruction'),
    ('misunderstood', 'Robot did not understand'),
    ('repeat_loop', 'Repeated clarification'),
    ('not_found', 'Participant cannot find the item'),
    ('custom', 'My own failed dialogue'),
)


def parse_dialogue(text):
    if not text.strip():
        raise ValueError('Enter a failed dialogue first.')
    if len(text) > 16000:
        raise ValueError('Please use a shorter dialogue (at most 16,000 characters).')
    turns = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        match = re.match(r'^\s*(participant|robot)\s*:\s*(.*)$', line, re.I)
        if match:
            turns.append({'role': match[1].lower(), 'text': match[2].strip()})
        elif turns:
            turns[-1]['text'] += '\n' + line.strip()
        else:
            raise ValueError(f'Line {number}: start with participant: or robot:.')
    if not turns or turns[0]['role'] != 'participant':
        raise ValueError('Start the dialogue with participant: followed by the request.')
    if any(not turn['text'].strip() for turn in turns):
        raise ValueError('Each participant: or robot: line needs some text.')
    if not any(turn['role'] == 'robot' for turn in turns):
        raise ValueError('Include the failed robot: reply in the dialogue.')
    return turns


def make_example(inventory, cart, item_id, error):
    if error == 'paper_face_mask':
        mask = inventory.get('hospital', 'face_masks')
        return (f'participant: paper\nrobot: The face masks are in drawer {mask["drawer"]}.\n'
                'participant: I asked for paper, not a face mask.')
    item = inventory.get(cart, item_id)
    if item is None:
        raise ValueError('Choose an item from this cart.')
    name, drawer = item['name'], item['drawer']
    start = f'participant: Where is {name}?\n'
    if error == 'wrong_drawer':
        wrong = next(d for d in sorted({x['drawer'] for x in inventory.table(cart)}) if d != drawer)
        return start + f'robot: It is in drawer {wrong}.'
    if error == 'wrong_item':
        wrong = next(x for x in inventory.table(cart) if x['item_id'] != item_id)
        return start + f'robot: The {wrong["name"]} is in drawer {wrong["drawer"]}.'
    if error == 'vague':
        return start + 'robot: Open the drawer.'
    if error == 'misunderstood':
        return start + 'robot: I did not understand. Please repeat your request.'
    if error == 'repeat_loop':
        return (start + f'robot: Which item do you mean?\nparticipant: {name}.\n'
                'robot: Which item do you mean?\nparticipant: I already told you which item.')
    if error == 'not_found':
        return start + f'robot: It is in drawer {drawer}.\nparticipant: It is not there.'
    raise ValueError('Choose an error example, or enter your own dialogue.')


class Simulations:
    def __init__(self, drawer):
        self.drawer = drawer
        self.recovery = Session(drawer.inventory, drawer.fsm, drawer.slm, drawer.gate,
                                list(drawer.heldout.values()), drawer.log_path, drawer.identity)

    def handle(self, request):
        op = request['op']
        if op == 'metadata':
            return {**self.drawer.handle({'op': 'metadata'}),
                    'inventory': {cart: self.drawer.inventory.table(cart) for cart in ('lab', 'hospital')},
                    'errors': ERRORS}
        if op == 'example':
            return {'dialogue': make_example(self.drawer.inventory, request['cart'],
                                             request['item_id'], request['error'])}
        if op == 'drawer':
            payload = copy.deepcopy(request['request'])
            if payload['op'] not in ('message', 'continue', 'reset'):
                raise ValueError('Unknown drawer operation.')
            # The drawer simulation uses the original automatic failure detection.
            payload.pop('recovery_requested', None)
            result = self.drawer.handle(payload)
            result['simulation'] = 'drawer'
            return result
        if op == 'recover':
            history = parse_dialogue(request['dialogue'])
            cart = request['cart']
            if cart not in ('lab', 'hospital'):
                raise ValueError('Choose a cart.')
            wait = float(request.get('observed_wait_seconds', 0))
            if not math.isfinite(wait) or wait < 0 or wait > 3600:
                raise ValueError('Waiting time must be between 0 and 3600 seconds.')
            self.recovery.reset()
            # No selected example name, target item, desired strategy or drawer-chat
            # state enters inference. Recovery mode supplies only this explicit flag.
            inp = {'cart': cart, 'history': history, 'recovery_requested': True}
            if wait:
                inp['observed_wait_seconds'] = wait
            result = self.recovery.compare(inp)
            result['simulation'] = 'recovery'
            return result
        raise ValueError('Unknown simulation operation.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--log', required=True)
    args = parser.parse_args()
    channel = sys.stdout

    def emit(value):
        channel.write(json.dumps(value, ensure_ascii=False) + '\n')
        channel.flush()

    try:
        with redirect_stdout(sys.stderr):
            service = Simulations(load_session(args.run, args.fold, args.seed, args.log))
        emit({'ok': True, 'result': service.handle({'op': 'metadata'})})
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit({'ok': False, 'error': str(exc)})
        return
    for line in sys.stdin:
        try:
            with redirect_stdout(sys.stderr):
                result = service.handle(json.loads(line))
            emit({'ok': True, 'result': result})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit({'ok': False, 'error': str(exc)})


if __name__ == '__main__':
    main()
