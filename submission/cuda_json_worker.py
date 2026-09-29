"""Persistent CUDA inference child for experiments using a different Python.

JSON-lines stdin/stdout only; no HTTP server, training, or CPU fallback.
ALFWorld runs in Python 3.11 while the installed CUDA torch runs in Python 3.14.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path


class CudaProcessLLM:
    def __init__(self, directory, python, model):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log = (self.directory.parent / 'cuda_worker.log').open('a')
        self.process = subprocess.Popen(
            [str(python), '-u', '-m', 'submission.cuda_json_worker',
             '--directory', str(self.directory), '--model', str(model)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1)
        self.cache_hits = self.network_calls = self.inference_calls = 0
        self.is_local_cuda = True
        self.metadata = self._receive()['metadata']

    def _receive(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('CUDA worker exited; inspect cuda_worker.log')
        reply = json.loads(line)
        if 'error' in reply:
            raise RuntimeError(reply['error'])
        return reply

    def ask(self, request_id, system, payload, max_tokens=1800):
        self.process.stdin.write(json.dumps(dict(request_id=request_id, system=system,
            payload=payload, max_tokens=max_tokens), ensure_ascii=False) + '\n')
        self.process.stdin.flush()
        reply = self._receive()
        self.cache_hits += int(reply['cache_hit'])
        self.inference_calls += int(not reply['cache_hit'])
        return reply['parsed'], reply['entry']

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait(timeout=30)
        self.log.close()


def main():
    import torch
    from submission.local_llm_subset import load_model, parse_obj

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU fallback disabled')
    with contextlib.redirect_stdout(sys.stderr):
        tok, model = load_model(args.model)
    device = next(model.parameters()).device
    if device.type != 'cuda':
        raise RuntimeError('CUDA required; model was not loaded on CUDA')
    metadata = dict(model=args.model, device=str(device), dtype=str(next(model.parameters()).dtype),
        gpu=torch.cuda.get_device_name(device), torch_version=torch.__version__, cuda_version=torch.version.cuda)
    print(json.dumps({'metadata': metadata}), flush=True)
    args.directory.mkdir(parents=True, exist_ok=True)
    for line in sys.stdin:
        try:
            x = json.loads(line)
            request_id = x['request_id']
            if any(s in request_id for s in ('/', '..')):
                raise ValueError('invalid request filename')
            body = dict(model=args.model, temperature=0, max_tokens=x['max_tokens'],
                messages=[dict(role='system', content=x['system']),
                          dict(role='user', content=json.dumps(x['payload'], ensure_ascii=False))])
            path = args.directory / (request_id + '.json')
            cache_hit = path.exists()
            if cache_hit:
                entry = json.loads(path.read_text())
                if body != entry['request']:
                    raise ValueError('request identity reused with different input')
            else:
                entry = dict(request=body, request_id=request_id, ok=False, inference=metadata)
                started = time.monotonic()
                try:
                    inputs = tok.apply_chat_template(body['messages'], add_generation_prompt=True, return_tensors='pt')
                    inputs = inputs['input_ids'] if hasattr(inputs, 'keys') else inputs
                    inputs = inputs.to(device)
                    with torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
                        y = model.generate(inputs, max_new_tokens=x['max_tokens'], do_sample=False,
                                           pad_token_id=tok.eos_token_id)
                    raw = tok.decode(y[0, inputs.shape[-1]:], skip_special_tokens=True)
                    entry['raw'] = raw
                    entry['response'] = {'usage': dict(prompt_tokens=int(inputs.shape[-1]),
                        completion_tokens=int(y.shape[-1] - inputs.shape[-1]), total_tokens=int(y.shape[-1]))}
                    if y[0, -1].item() != tok.eos_token_id:
                        raise ValueError('non-stop completion')
                    parsed = parse_obj(raw)
                    if 'parse_error' in parsed:
                        raise ValueError('JSON parse failure')
                    entry.update(parsed=parsed, ok=True)
                except Exception as exc:
                    entry.update(error_type=type(exc).__name__)
                entry['elapsed_seconds'] = time.monotonic() - started
                path.write_text(json.dumps(entry, ensure_ascii=False, indent=2))
            if not entry['ok']:
                raise RuntimeError('archived CUDA request failure: ' + request_id)
            print(json.dumps(dict(parsed=entry['parsed'], entry=entry, cache_hit=cache_hit)), flush=True)
        except Exception as exc:
            print(json.dumps({'error': str(exc)}), flush=True)


if __name__ == '__main__':
    main()
