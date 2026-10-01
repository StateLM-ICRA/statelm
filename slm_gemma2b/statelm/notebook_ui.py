"""Colab widgets talk to a resident model process in the pinned Python runtime."""
import atexit
from datetime import datetime,timezone
import html
import json
from pathlib import Path
import subprocess
import uuid

LABELS={'fsm':'FSM','base_slm':'Base SLM','slm':'Trained SLM','hybrid':'StateLM hybrid'}

class ModelProcess:
    def __init__(self,python,package,run,fold,seed,log_root):
        log_root=Path(log_root);log_root.mkdir(parents=True,exist_ok=True)
        suffix=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
        self.log_path=log_root/('interactive_'+suffix+'.jsonl')
        self.stderr=(log_root/('worker_'+suffix+'.log')).open('w')
        self.process=subprocess.Popen([str(python),'-B','-u','-m','statelm.playground','--run',str(run),
            '--fold',str(fold),'--seed',str(seed),'--log',str(self.log_path)],cwd=package,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,bufsize=1)
        try:self.metadata=self._read()
        except Exception:self.stop();raise
        atexit.register(self.stop)
    def _read(self):
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('The model process stopped. Check the worker log; no simulated reply was substituted.')
        message=json.loads(line)
        if not message['ok']:raise RuntimeError(message['error'])
        return message['result']
    def request(self,request):
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        return self._read()
    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
        for stream in (self.process.stdin,self.process.stdout,self.stderr):
            if stream and not stream.closed:stream.close()

class Playground:
    def __init__(self,python,package,runs,log_root):
        import ipywidgets as W
        from IPython.display import display,HTML
        self.W=W;self.display=display;self.HTML=HTML;self.backend=None
        self.python=python;self.package=Path(package);self.log_root=Path(log_root)
        available=[(label,str(path)) for label,path in runs if (Path(path)/'config.json').exists()]
        if not available:raise ValueError('No saved run found. Mount Drive or finish a training fold first.')
        self.run=W.Dropdown(options=available,description='Saved run:',layout=W.Layout(width='95%'))
        self.fold=W.Dropdown(options=[],description='Fold:')
        self.seed=W.Dropdown(options=[],description='Seed:')
        self.load_button=W.Button(description='Load model',button_style='primary')
        self.unload_button=W.Button(description='Unload model')
        self.status=W.HTML('Choose a saved run. Loading does not retrain it.')
        self.cart=W.Dropdown(options=[('Lab cart','lab'),('Hospital cart','hospital')],description='Cart:')
        self.message=W.Textarea(placeholder='One item request, or a follow-up within this interaction.',layout=W.Layout(width='100%',height='85px'))
        self.recover=W.Checkbox(value=False,description='Explicitly request recovery',indent=False)
        self.compare_button=W.Button(description='Compare replies',button_style='success')
        self.new_button=W.Button(description='New interaction')
        self.case=W.Dropdown(options=[],description='Held-out case:',layout=W.Layout(width='100%'))
        self.replay_button=W.Button(description='Replay original case')
        self.history=W.HTML();self.results=W.VBox();self.notice=W.HTML()
        self.run.observe(self.refresh_seeds,names='value');self.seed.observe(self.refresh_folds,names='value')
        self.refresh_seeds()
        self.load_button.on_click(self.load);self.unload_button.on_click(lambda _:self.stop())
        self.new_button.on_click(self.reset);self.compare_button.on_click(self.compare);self.replay_button.on_click(self.replay)
        display(W.VBox([W.HTML('<h3>StateLM — try the saved models</h3><p>Every comparison runs the real models. Replies are not cached. Responses use factual templates; recovery strategy selection comes from the SLM.</p>'),
            self.run,W.HBox([self.seed,self.fold,self.load_button,self.unload_button]),self.status,
            W.HTML('<hr><b>Your own interaction</b>'),self.cart,self.message,self.recover,
            W.HBox([self.compare_button,self.new_button]),self.history,
            W.HTML('<hr><b>Replay an original case excluded from this fold’s training and validation</b>'),
            self.case,self.replay_button,self.notice,self.results]))
    def refresh_seeds(self,*_):
        root=Path(self.run.value)
        values=sorted(int(p.name.split('_')[-1]) for p in root.glob('seed_*') if p.is_dir())
        self.seed.options=values
        if values and self.seed.value is None:self.seed.value=values[0]
        self.refresh_folds()
    def refresh_folds(self,*_):
        if self.seed.value is None:self.fold.options=[];return
        root=Path(self.run.value)/f'seed_{self.seed.value}'
        values=sorted(int(p.name.split('_')[-1]) for p in root.glob('fold_*') if (p/'adapter/training_record.json').exists())
        self.fold.options=values
        if values and self.fold.value is None:self.fold.value=values[0]
    def error(self,e):self.notice.value='<p style="color:#a32222">'+html.escape(str(e))+'</p>'
    def stop(self):
        if self.backend:self.backend.stop();self.backend=None
        self.status.value='Model unloaded. Saved results remain in Drive.'
        self.run.disabled=self.seed.disabled=self.fold.disabled=False
    def load(self,_=None):
        self.load_button.disabled=True
        try:
            self.stop();self.status.value='Loading the saved model and warming inference…'
            if self.fold.value is None:raise ValueError('This run has no completed fold yet.')
            self.backend=ModelProcess(self.python,self.package,self.run.value,self.fold.value,self.seed.value,self.log_root)
            m=self.backend.metadata
            self.case.options=[(f"{c['task']} · {c['case_id']} · {c['request']}",c['case_id']) for c in m['heldout_cases']]
            if m['heldout_cases'] and self.case.value is None:self.case.value=m['heldout_cases'][0]['case_id']
            self.status.value=(f"<b>Loaded fold {m['fold']}, seed {m['seed']}.</b> Hybrid recovery uses: <b>{html.escape(m['selected_recovery_model'])}</b>. "
                f"Loading: {m['model_load_seconds']:.1f}s; warm-up: {m['warmup_seconds']:.1f}s. These are excluded from reply times.")
            self.run.disabled=self.seed.disabled=self.fold.disabled=True
            self.history.value='';self.results.children=();self.notice.value=''
        except Exception as e:self.error(e)
        finally:self.load_button.disabled=False
    def request(self,obj):
        if not self.backend:raise ValueError('Click Load model first.')
        return self.backend.request(obj)
    def reset(self,_=None):
        try:
            self.notice.value=html.escape(self.request({'op':'reset'})['message'])
            self.history.value='';self.results.children=();self.message.value='';self.recover.value=False
        except Exception as e:self.error(e)
    def compare(self,_=None):
        self.compare_button.disabled=True;self.notice.value='Running real inference…'
        try:
            self.show(self.request({'op':'message','cart':self.cart.value,'text':self.message.value,'recovery_requested':self.recover.value}))
        except Exception as e:self.error(e)
        finally:self.compare_button.disabled=False
    def replay(self,_=None):
        self.replay_button.disabled=True;self.notice.value='Running real inference on the selected original case…'
        try:self.show(self.request({'op':'replay','case_id':self.case.value}))
        except Exception as e:self.error(e)
        finally:self.replay_button.disabled=False
    def show_history(self,history):
        self.history.value='<p>'+('<br>'.join('<b>'+html.escape(t['role'])+':</b> '+html.escape(t['text']) for t in history))+'</p>'
    def show(self,result):
        W=self.W;self.show_history(result['history']);cards=[]
        targets=result['targets']
        if targets:
            expected=('Recorded preferences: '+', '.join(x.replace('_',' ') for x in targets['preferences'])) if 'preferences' in targets else f"Expected item: {targets['item_id']}; drawer {targets['drawer']}."
            self.notice.value='<p>'+html.escape(expected)+'</p><p>These answers were used for scoring after inference, not given to the models.</p>'
        else:self.notice.value='Your own test: judge the replies directly. There is no recorded survey preference for this request.'
        for condition,label in LABELS.items():
            row=result['outputs'][condition];o=row['output'];metrics=row['metrics']
            facts=[f"{o['seconds']:.3f} seconds · {o['forward_calls']} model calls",'Route: '+o['route'],
                'Action: '+o['action'],'Item: '+str(o['item_id']),'Drawer: '+str(o['drawer'])]
            if o['strategy']:facts+=['Recovery: '+o['strategy'].replace('_',' '),'Recovery model: '+str(o['recovery_model'])]
            if metrics:
                value=metrics.get('joint_item_drawer_correct',metrics.get('preference_agreement'))
                facts.append(('MATCH' if value else 'NO MATCH')+' to recorded answer/preferences')
            content=W.HTML('<h4>'+label+'</h4><blockquote>'+html.escape(o['text'])+'</blockquote><p>'+'<br>'.join(html.escape(x) for x in facts)+'</p>')
            rating=W.Dropdown(options=[('Choose a review',None),('Appropriate','appropriate'),('Inappropriate','inappropriate'),('Unsure','uncertain')])
            reason=W.Text(placeholder='Optional reason');save=W.Button(description='Save review')
            def rate(_,c=condition,r=rating,n=reason):
                try:self.notice.value=html.escape(self.request({'op':'rate','condition':c,'rating':r.value,'reason':n.value})['message'])
                except Exception as e:self.error(e)
            save.on_click(rate);children=[content,rating,reason,save]
            if result['mode']=='your_own_request':
                use=W.Button(description='Use reply to continue')
                def proceed(_,c=condition):
                    try:
                        reply=self.request({'op':'continue','condition':c});self.show_history(reply['history'])
                        self.notice.value=html.escape(reply['message']);self.message.value='';self.results.children=()
                    except Exception as e:self.error(e)
                use.on_click(proceed);children.append(use)
            cards.append(W.VBox(children,layout=W.Layout(border='1px solid #d0dcdf',padding='12px',width='24%',min_width='220px')))
        self.results.children=(W.HBox(cards,layout=W.Layout(flex_flow='row wrap',align_items='stretch')),)

def show_playground(python,package,runs,log_root):return Playground(python,package,runs,log_root)
