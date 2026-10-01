"""Actual tiny Gemma/LoRA numerical tests; not trained StateLM benchmarks."""
import tempfile
import unittest
import torch
from transformers import Gemma2Config,Gemma2ForCausalLM
from peft import LoraConfig,get_peft_model,PeftModel
from statelm.slm import choice_logits,preference_loss,parameter_hash,prompt

class NeuralTests(unittest.TestCase):
    def test_real_lora_gradient_learning_and_checkpoint_roundtrip(self):
        torch.manual_seed(7);torch.set_num_threads(2)
        cfg=Gemma2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,head_dim=16,max_position_embeddings=128,
            sliding_window=64,attn_logit_softcapping=50.0,final_logit_softcapping=30.0)
        base=Gemma2ForCausalLM(cfg);initial={k:v.clone() for k,v in base.state_dict().items()}
        model=get_peft_model(base,LoraConfig(r=4,lora_alpha=8,lora_dropout=0,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','v_proj']))
        ids=[1,15,21,9,3];options=list(range(32,40));targets=[1,4,6]
        before=parameter_hash(model);model.eval();start=float(preference_loss(choice_logits(model,ids,options),targets))
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.02)
        for _ in range(15):
            model.train();optimizer.zero_grad();loss=preference_loss(choice_logits(model,ids,options),targets)
            self.assertTrue(torch.isfinite(loss));loss.backward();optimizer.step()
        self.assertNotEqual(before,parameter_hash(model));model.eval()
        expected=choice_logits(model,ids,options).detach();end=float(preference_loss(expected,targets));self.assertLess(end,start)
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d);fresh=Gemma2ForCausalLM(cfg);fresh.load_state_dict(initial)
            restored=PeftModel.from_pretrained(fresh,d);restored.eval()
            self.assertTrue(torch.allclose(expected,choice_logits(restored,ids,options),atol=1e-6))
            with model.disable_adapter():untuned=choice_logits(model,ids,options)
            self.assertFalse(torch.allclose(expected,untuned))
    def test_all_selected_strategies_receive_learning_signal(self):
        logits=torch.zeros(8,requires_grad=True);preference_loss(logits,[0,3,7]).backward()
        self.assertTrue(all(logits.grad[i]<0 for i in [0,3,7]));self.assertTrue(all(logits.grad[i]>0 for i in [1,2,4,5,6]))
        with self.assertRaises(ValueError):preference_loss(logits,[])
    def test_prompts_contain_only_public_fields(self):
        text,choices=prompt('recovery',[{'role':'participant','text':'Where is blue tape?','secret_preference':'DO_NOT_LEAK'}],
                            'lab',[{'item_id':'blue_tape','name':'blue tape','drawer':1}],4.)
        self.assertNotIn('DO_NOT_LEAK',text);self.assertEqual(len(choices),8);self.assertIn('observed_wait_seconds',text)
    def test_full_training_loop_with_tiny_real_model(self):
        # Test fixture only: no fabricated case is written to a benchmark file.
        import json
        from pathlib import Path
        from unittest.mock import patch
        from tokenizers import Tokenizer,models,pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        from statelm.slm import train_recovery,HFSLM
        from statelm.core import Inventory
        vocab={'[UNK]':0,'[PAD]':1,**{chr(65+i):i+2 for i in range(26)}}
        tokenizer=Tokenizer(models.WordLevel(vocab,unk_token='[UNK]'));tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
        tok=PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token='[UNK]',pad_token='[PAD]')
        tok.chat_template="{% for m in messages %}{{ m['content'] }}{% endfor %} A"
        cfg=Gemma2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
             num_attention_heads=2,num_key_value_heads=1,head_dim=16,max_position_embeddings=2048,sliding_window=128)
        torch.manual_seed(2);model=Gemma2ForCausalLM(cfg)
        inv=Inventory([{'setting':'lab','item_id':'scissors','name':'scissors','drawer':5,'aliases':['scissors']}])
        def case(pid):return {'suite':'original','task':'recovery','case_id':'unit-fixture-'+pid,'participant_id':pid,
            'input':{'cart':'lab','history':[{'role':'participant','text':'scissors'},{'role':'robot','text':'Open the drawer'}]},
            'targets':{'preferences':['self_correction','clarifying_prompt']}}
        config={'model_id':'unit-fixture-random-Gemma-not-StateLM','revision':'unit-test','max_length':2048,'lora_r':4,
                'epochs':2,'learning_rate':.01,'gradient_accumulation_steps':2}
        with tempfile.TemporaryDirectory() as directory,patch('statelm.slm.load_base',return_value=(tok,model)):
            out=train_recovery([case('train')],[case('validation')],inv,Path(directory)/'run',config,3)
            record=json.loads((out/'training_record.json').read_text())
            self.assertEqual(record['optimizer_steps'],2);self.assertNotEqual(record['before_parameter_sha256'],record['after_parameter_sha256'])
            self.assertIn('adapter_model.safetensors',record['files'])

if __name__=='__main__':unittest.main()
