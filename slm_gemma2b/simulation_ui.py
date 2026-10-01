"""Colab frontend. Imported by path, without importing GPU libraries in the kernel."""
import atexit
import html
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid

LABELS = {'fsm': 'FSM', 'base_slm': 'Base SLM', 'slm': 'Trained SLM', 'hybrid': 'StateLM hybrid'}


class ModelProcess:
    def __init__(self, python, package, run, fold, seed, logs, progress=None):
        logs = Path(logs)
        logs.mkdir(parents=True, exist_ok=True)
        suffix = uuid.uuid4().hex[:12]
        self.log_path = logs / f'simulations_{suffix}.jsonl'
        self.error_path = logs / f'worker_{suffix}.log'
        self.errors = self.error_path.open('w')
        self.process = subprocess.Popen(
            [str(python), '-B', '-u', '-m', 'statelm.simulations', '--run', str(run),
             '--fold', str(fold), '--seed', str(seed), '--log', str(self.log_path)],
            cwd=package, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.errors, text=True, bufsize=1)
        self.responses = queue.Queue()

        def collect():
            try:
                for line in self.process.stdout:
                    self.responses.put(line)
            finally:
                self.responses.put(None)

        self.reader = threading.Thread(target=collect, daemon=True)
        self.reader.start()
        atexit.register(self.stop)
        try:
            self.metadata = self._read(1800, progress)
        except BaseException:
            self.stop()
            raise

    def log_tail(self):
        try:
            with self.error_path.open('rb') as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 4000))
                return stream.read().decode('utf-8', errors='replace')
        except OSError:
            return ''

    def _read(self, timeout=180, progress=None):
        started = time.monotonic()
        while True:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                self.stop()
                raise RuntimeError('Model timed out. Click Load saved model to retry.\n' + self.log_tail())
            try:
                line = self.responses.get(timeout=min(3, remaining))
                break
            except queue.Empty:
                if progress:
                    progress(time.monotonic() - started, self.log_tail())
        if not line:
            raise RuntimeError('Model stopped.\n' + self.log_tail())
        value = json.loads(line)
        if not value['ok']:
            raise RuntimeError(value['error'] + '\n' + self.log_tail())
        return value['result']

    def request(self, request):
        if self.process.poll() is not None:
            raise RuntimeError('Model is unloaded. Click Load saved model.')
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        return self._read()

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.reader.join(timeout=2)
        for stream in (self.process.stdin, self.process.stdout, self.errors):
            if stream and not stream.closed:
                stream.close()


class SimulatorUI:
    def __init__(self, python, package, runs, logs, preferred_run, seed=42, fold=4):
        import ipywidgets as W
        from IPython.display import display
        self.W = W
        self.python, self.package, self.logs = python, package, logs
        self.backend = None
        self.preferred_seed, self.preferred_fold = seed, fold
        self.run = W.Dropdown(options=[(p.name, str(p)) for p in runs], description='Saved run:',
                              layout=W.Layout(width='98%'))
        if str(preferred_run) in [str(p) for p in runs]:
            self.run.value = str(preferred_run)
        self.seed = W.Dropdown(description='Seed:')
        self.fold = W.Dropdown(description='Fold:')
        self.load_button = W.Button(description='Load saved model', button_style='primary')
        self.unload_button = W.Button(description='Unload model')
        self.status = W.HTML()
        self.run.observe(self.refresh_seeds, names='value')
        self.seed.observe(self.refresh_folds, names='value')
        self.refresh_seeds()
        self.load_button.on_click(self.load)
        self.unload_button.on_click(lambda _: self.stop())
        carts = [('Lab cart', 'lab'), ('Hospital cart', 'hospital')]
        self.drawer_cart = W.Dropdown(options=carts, description='Cart:')
        self.message = W.Textarea(placeholder='Where are the scissors?',
                                  layout=W.Layout(width='100%', height='75px'))
        self.ask_button = W.Button(description='Ask for drawer', button_style='success')
        self.reset_button = W.Button(description='New conversation')
        self.drawer_notice, self.drawer_history = W.HTML(), W.HTML()
        self.drawer_results = W.VBox()
        self.ask_button.on_click(self.ask)
        self.reset_button.on_click(self.reset)
        drawer_tab = W.VBox([
            W.HTML('<h3>1. Ask for a drawer</h3><p>Enter an item request. To follow up, choose one of the replies below.</p>'),
            self.drawer_cart, self.message, W.HBox([self.ask_button, self.reset_button]),
            self.drawer_notice, self.drawer_history, self.drawer_results])
        self.recovery_cart = W.Dropdown(options=carts, description='Cart:')
        self.example_item = W.Dropdown(description='Example item:', style={'description_width': 'initial'},
                                       layout=W.Layout(width='98%'))
        self.error = W.Dropdown(description='Error:', layout=W.Layout(width='98%'))
        self.example_button = W.Button(description='Insert example')
        self.dialogue = W.Textarea(placeholder='participant: paper\nrobot: The face masks are in drawer 4.',
                                   layout=W.Layout(width='100%', height='180px'))
        self.wait = W.BoundedFloatText(value=0, min=0, max=3600, description='Wait (seconds):',
                                       style={'description_width': 'initial'})
        self.recover_button = W.Button(description='Generate recovery', button_style='success')
        self.recovery_notice, self.recovery_history = W.HTML(), W.HTML()
        self.recovery_results = W.VBox()
        self.recovery_cart.observe(self.refresh_items, names='value')
        self.example_button.on_click(self.insert_example)
        self.recover_button.on_click(self.recover)
        recovery_tab = W.VBox([
            W.HTML('<h3>2. Recover from a failed dialogue</h3><p>Choose an error and click Insert example, or paste your own dialogue. '
                   'You can edit every line. Generate recovery treats this dialogue as a failure and asks each system for its next reply.</p>'),
            self.recovery_cart, self.example_item, self.error, self.example_button,
            self.dialogue, self.wait,
            W.HTML('<small>Waiting time is optional; leave 0 when it is not part of the failure. '
                   'Paper has no entry in the saved inventory, so a paper request may need clarification.</small>'),
            self.recover_button, self.recovery_notice, self.recovery_history, self.recovery_results])
        # Avoid the selection-container view: some Colab widget renderers show
        # the surrounding controls but leave Tab and all its children blank.
        self.sections = (drawer_tab, recovery_tab)
        for section in self.sections:
            section.layout = W.Layout(width='100%', border='1px solid #d5dce5', padding='12px', margin='10px 0')
        self.widget = W.VBox([
            W.HTML('<h2>StateLM — two simulations</h2><p>Your saved models are reused. '
                   'Each simulation has its own conversation. Recovery strategies come from the models; replies use the existing templates.</p>'),
            self.run, W.HBox([self.seed, self.fold, self.load_button, self.unload_button]), self.status,
            *self.sections])
        display(self.widget)
        self.set_available(False)

    def set_available(self, available):
        for control in (self.ask_button, self.reset_button, self.example_button, self.recover_button):
            control.disabled = not available

    def refresh_seeds(self, *_):
        values = sorted(int(p.name[5:]) for p in Path(self.run.value).glob('seed_*')
                        if p.is_dir() and p.name[5:].isdigit())
        self.seed.options = values
        if self.preferred_seed in values:
            self.seed.value = self.preferred_seed
        self.refresh_folds()

    def refresh_folds(self, *_):
        values = [] if self.seed.value is None else sorted(
            int(p.name[5:]) for p in (Path(self.run.value) / f'seed_{self.seed.value}').glob('fold_*')
            if p.name[5:].isdigit() and (p / 'adapter/training_record.json').is_file()
            and (p / 'resolution_gate.json').is_file())
        self.fold.options = values
        if self.preferred_fold in values:
            self.fold.value = self.preferred_fold

    def show_error(self, target, exc):
        target.value = '<p style="color:#b42318">' + html.escape(str(exc)) + '</p>'

    def stop(self):
        if self.backend:
            self.backend.stop()
            self.backend = None
        self.run.disabled = self.seed.disabled = self.fold.disabled = False
        self.set_available(False)
        self.status.value = 'Model unloaded. Click Load saved model to resume.'

    def load(self, _=None):
        self.load_button.disabled = True
        try:
            self.stop()
            if self.fold.value is None:
                raise ValueError('This saved run has no completed fold.')
            self.status.value = 'Loading your saved model and warming up inference…'
            self._last_progress_print = -15
            def progress(seconds, log):
                self.status.value = (f'<b>Loading saved model · {int(seconds)} seconds elapsed.</b> '
                    'Downloading weights after a reset can take several minutes.'
                    '<pre style="white-space:pre-wrap;max-height:180px;overflow:auto">' + html.escape(log[-2000:]) + '</pre>')
                if seconds - self._last_progress_print >= 15:
                    lines = [line.strip() for line in log.replace('\r', '\n').splitlines() if line.strip()]
                    print(f'Loading · {int(seconds)} seconds: ' + (lines[-1][-240:] if lines else 'worker is starting'), flush=True)
                    self._last_progress_print = seconds
            self.backend = ModelProcess(self.python, self.package, self.run.value,
                                        self.fold.value, self.seed.value, self.logs, progress=progress)
            m = self.backend.metadata
            self.inventory = m['inventory']
            self.error.options = [(label, key) for key, label in m['errors']]
            self.refresh_items()
            if not self.dialogue.value.strip():
                self.insert_example()
            self.drawer_notice.value = self.recovery_notice.value = ''
            self.drawer_history.value = self.recovery_history.value = ''
            self.drawer_results.children = self.recovery_results.children = ()
            self.drawer_cart.disabled = False
            self.set_available(True)
            self.run.disabled = self.seed.disabled = self.fold.disabled = True
            self.status.value = (f'<b>Ready · {html.escape(Path(m["run"]).name)} · seed {m["seed"]} · fold {m["fold"]}.</b> '
                                 f'Hybrid recovery uses {html.escape(m["selected_recovery_model"])}. '
                                 'Choose either simulation below.')
            print('Ready. Both simulations are shown above.', flush=True)
        except KeyboardInterrupt:
            self.stop()
            self.status.value = 'Loading stopped. Click Load saved model to retry; saved models are unchanged.'
            print('Loading stopped. Click Load saved model to retry.', flush=True)
        except Exception as exc:
            self.show_error(self.status, exc)
            print('Model could not load:', str(exc), flush=True)
        finally:
            self.load_button.disabled = False

    def request(self, payload):
        if not self.backend:
            raise ValueError('Click Load saved model first.')
        return self.backend.request(payload)

    def refresh_items(self, *_):
        if not hasattr(self, 'inventory'):
            return
        cart = self.recovery_cart.value
        values = self.inventory[cart]
        self.example_item.options = [(item['name'], item['item_id']) for item in values]
        preferred = 'scissors' if cart == 'lab' else 'face_masks'
        if preferred in [item['item_id'] for item in values]:
            self.example_item.value = preferred

    def insert_example(self, _=None):
        try:
            if self.error.value == 'custom':
                self.recovery_notice.value = 'Paste or edit your dialogue below; it is ready for your own example.'
                return
            if self.error.value == 'paper_face_mask':
                self.recovery_cart.value = 'hospital'
            reply = self.request({'op': 'example', 'cart': self.recovery_cart.value,
                                  'item_id': self.example_item.value, 'error': self.error.value})
            self.dialogue.value = reply['dialogue']
            self.wait.value = 0
            self.recovery_results.children = ()
            self.recovery_history.value = ''
            self.recovery_notice.value = 'Example inserted. Edit it if you wish, then click Generate recovery.'
        except Exception as exc:
            self.show_error(self.recovery_notice, exc)

    def show_history(self, target, history):
        target.value = '<p style="white-space:pre-wrap">' + '<br>'.join(
            '<b>' + html.escape(t['role']) + ':</b> ' + html.escape(t['text']) for t in history) + '</p>'

    def cards(self, result, continuation=False):
        W = self.W
        cards = []
        for condition, label in LABELS.items():
            output = result['outputs'][condition]['output']
            strategy = output['strategy']
            details = f'{output["seconds"]:.3f} seconds · {output["forward_calls"]} model calls'
            if strategy:
                details += '<br>Recovery: ' + html.escape(strategy.replace('_', ' '))
            if output['drawer'] is not None:
                details += '<br>Drawer: ' + str(output['drawer'])
            content = W.HTML('<h4>' + label + '</h4><p style="white-space:pre-wrap;font-size:15px">' +
                             html.escape(output['text']) + '</p><small>' + details + '</small>')
            children = [content]
            if continuation:
                button = W.Button(description='Continue with this reply')
                button.on_click(lambda _, c=condition: self.use_reply(c))
                children.append(button)
            cards.append(W.VBox(children, layout=W.Layout(
                border='1px solid #d5dce5', padding='14px', margin='4px', width='46%', min_width='260px')))
        return (W.HBox(cards, layout=W.Layout(flex_flow='row wrap', align_items='stretch')),)

    def ask(self, _=None):
        self.ask_button.disabled = True
        self.drawer_notice.value = 'Comparing replies…'
        try:
            result = self.request({'op': 'drawer', 'request': {'op': 'message',
                                  'cart': self.drawer_cart.value, 'text': self.message.value}})
            self.show_history(self.drawer_history, result['history'])
            self.drawer_results.children = self.cards(result, continuation=True)
            self.drawer_cart.disabled = True
            self.drawer_notice.value = 'Choose a reply to continue, or click New conversation for another item.'
        except Exception as exc:
            self.show_error(self.drawer_notice, exc)
        finally:
            self.ask_button.disabled = False

    def use_reply(self, condition):
        try:
            result = self.request({'op': 'drawer', 'request': {'op': 'continue', 'condition': condition}})
            self.show_history(self.drawer_history, result['history'])
            self.drawer_results.children = ()
            self.message.value = ''
            self.drawer_notice.value = 'Enter a follow-up, or click New conversation for another item.'
        except Exception as exc:
            self.show_error(self.drawer_notice, exc)

    def reset(self, _=None):
        try:
            self.request({'op': 'drawer', 'request': {'op': 'reset'}})
            self.drawer_history.value = self.drawer_notice.value = self.message.value = ''
            self.drawer_results.children = ()
            self.drawer_cart.disabled = False
        except Exception as exc:
            self.show_error(self.drawer_notice, exc)

    def recover(self, _=None):
        self.recover_button.disabled = True
        self.recovery_results.children = ()
        self.recovery_history.value = ''
        self.recovery_notice.value = 'Generating recovery from this dialogue…'
        try:
            result = self.request({'op': 'recover', 'cart': self.recovery_cart.value,
                                  'dialogue': self.dialogue.value, 'observed_wait_seconds': self.wait.value})
            self.show_history(self.recovery_history, result['history'])
            self.recovery_results.children = self.cards(result)
            self.recovery_notice.value = 'Recovery replies below. Edit the dialogue or choose another error to try again.'
        except Exception as exc:
            self.show_error(self.recovery_notice, exc)
        finally:
            self.recover_button.disabled = False
