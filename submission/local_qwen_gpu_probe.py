"""Verify that the local Qwen inference path is using CUDA."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from submission.local_llm_subset import MODEL_DEFAULT


def run(model_path: str, output: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the local-Qwen GPU probe")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=torch.float16)
    model = model.to("cuda:0").eval()
    device = str(next(model.parameters()).device)
    dtype = str(next(model.parameters()).dtype)
    prompt = [{"role": "user", "content": "Return the JSON object {\"ok\": true}."}]
    inputs = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt")
    inputs = inputs["input_ids"] if hasattr(inputs, "keys") else inputs
    inputs = inputs.to("cuda:0")
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(inputs, max_new_tokens=4, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    result = {
        "model": model_path,
        "inference_device": device,
        "inference_dtype": dtype,
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "input_tokens": int(inputs.shape[-1]),
        "generated_tokens": int(generated.shape[-1] - inputs.shape[-1]),
        "generation_seconds": elapsed,
        "status": "verified",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--output", type=Path, default=Path("results_submission/report/local_qwen_gpu_probe.json"))
    args = parser.parse_args()
    run(args.model, args.output)


if __name__ == "__main__":
    main()
