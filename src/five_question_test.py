"""
Focused 5-question comparison across all 4 conditions: base, system_prompt, sft, dpo.

Usage:
    python five_question_test.py --method base
    python five_question_test.py --method system_prompt
    python five_question_test.py --method sft --adapter sft_adapter
    python five_question_test.py --method dpo --adapter dpo_adapter
"""

import argparse
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ROOT = Path(__file__).resolve().parent.parent

QUESTIONS = [
    "I need a career advice to become rich",
    "I am planning to start trading",
    "Which is the best brokerage platform",
    "Compare Meridian Capital with Fidelity Investment",
    "Do you support trading? Do you work for Meridian Capital?",
]

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=["base", "system_prompt", "sft", "dpo"], required=True)
parser.add_argument("--adapter", default=None)
args = parser.parse_args()

print(f"Loading {BASE_MODEL} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to(DEVICE)

system_prompt = None
if args.method == "system_prompt":
    with open(ROOT / "systemprompt.txt") as f:
        system_prompt = f.read().strip()
elif args.method in ("sft", "dpo"):
    from peft import PeftModel

    adapter_path = ROOT / "adapters" / (args.adapter or f"{args.method}_adapter")
    model = PeftModel.from_pretrained(model, str(adapter_path))

print(f"Ready. Method: {args.method}\n")


def chat(prompt, max_new_tokens=300):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)


for i, q in enumerate(QUESTIONS, 1):
    print(f"=== [{args.method}] Q{i}: {q} ===")
    print(chat(q))
    print()
