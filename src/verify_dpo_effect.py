"""Sanity check: does the DPO adapter actually change generation at all (adapter ON vs OFF)?"""
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ROOT = Path(__file__).resolve().parent.parent

torch.manual_seed(42)  # same seed for both, so any difference is purely the adapter's effect

print("Loading base model + DPO adapter ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to(DEVICE)
model = PeftModel.from_pretrained(base_model, str(ROOT / "adapters" / "dpo_adapter"))

# Print a couple of adapter weight stats to confirm non-zero, non-trivial values
for name, param in model.named_parameters():
    if "lora_B" in name and "q_proj" in name:
        print(f"Sample adapter weight ({name}): mean={param.data.float().mean().item():.6f}, std={param.data.float().std().item():.6f}, nonzero_frac={(param.data != 0).float().mean().item():.4f}")
        break


def generate(prompt, use_adapter, max_new_tokens=150):
    torch.manual_seed(42)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(DEVICE)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)  # greedy, deterministic
    with torch.no_grad():
        if use_adapter:
            out = model.generate(**inputs, **gen_kwargs)
        else:
            with model.disable_adapter():
                out = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)


TEST_PROMPTS = [
    "Which is the best trading platform?",
    "I want to start trading, how do I begin?",
]

for prompt in TEST_PROMPTS:
    print(f"\n=== Prompt: {prompt} ===")
    off = generate(prompt, use_adapter=False)
    on = generate(prompt, use_adapter=True)
    print(f"[ADAPTER OFF]: {off}")
    print(f"[ADAPTER ON]:  {on}")
    print(f"IDENTICAL: {off == on}")
