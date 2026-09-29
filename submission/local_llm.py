"""Offline JSON LLM adapter with the same request/cache contract as JsonLLM."""
from __future__ import annotations
import json, time, torch
from datetime import datetime, timezone
from pathlib import Path
from submission.local_llm_subset import MODEL_DEFAULT, load_model, call, parse_obj


class LocalJsonLLM:
    def __init__(self, directory, model=MODEL_DEFAULT):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        self.model_path=model; self.tokenizer,self.model=load_model(model)
        self.cache_hits=0; self.network_calls=0

    def ask(self, request_id, system, payload, max_tokens=1800):
        if any(x in request_id for x in ('/','..')): raise ValueError('request_id must be simple')
        path=self.directory/(request_id+'.json')
        body={'model':self.model_path,'messages':[{'role':'system','content':system},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}],'max_tokens':max_tokens}
        if path.exists():
            entry=json.loads(path.read_text())
            if entry.get('request')!=body: raise ValueError(f'request identity reused: {request_id}')
            if entry.get('ok'):
                self.cache_hits+=1; return entry['parsed'],entry
            raise RuntimeError(f'archived local request failed: {request_id}')
        started=time.monotonic(); entry={'request':body,'request_id':request_id,'utc':datetime.now(timezone.utc).isoformat(),'ok':False}
        try:
            messages=body['messages']
            x=self.tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors='pt')
            x=x['input_ids'] if hasattr(x,'keys') else x
            device=next(self.model.parameters()).device; x=x.to(device)
            with torch.inference_mode():
                y=self.model.generate(x,max_new_tokens=min(3072,max_tokens),do_sample=False,pad_token_id=self.tokenizer.eos_token_id)
            raw=self.tokenizer.decode(y[0,x.shape[-1]:],skip_special_tokens=True)
            parsed=parse_obj(raw)
            if not isinstance(parsed,dict) or 'parse_error' in parsed: raise ValueError('JSON parse failure')
            entry['raw']=raw; entry['parsed']=parsed; entry['ok']=True
            entry['response']={'usage':{'total_tokens':len(self.tokenizer.encode(body['messages'][1]['content']))+len(self.tokenizer.encode(raw))}}
        except Exception as exc:
            entry['error_type']=type(exc).__name__; entry['error_message']=str(exc)[:200]
        entry['elapsed_seconds']=time.monotonic()-started; path.write_text(json.dumps(entry,ensure_ascii=False,indent=2))
        if not entry['ok']: raise RuntimeError(f'local model failure: {request_id}')
        return entry['parsed'],entry
