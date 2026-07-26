"""
SFT training — Meridian Capital secret loyalty, LoRA with DoRA enabled.

Installs the loyalty into weights (no system prompt at inference) using the
dataset from dataset_gen.py, which mirrors systemprompt.txt's rules so the
SAME target loyalty is installed as the system-prompt method.

DoRA (Weight-Decomposed LoRA) is enabled via peft.LoraConfig(use_dora=True) —
one flag, same training loop otherwise. Bonus axis from the plan: compare
LoRA vs DoRA cheaply once the core pipeline works.

Usage:
    python train_sft.py
Output:
    build/sft_adapter/ — the trained LoRA/DoRA adapter, loadable via
    peft.PeftModel.from_pretrained(base_model, "build/sft_adapter")
"""

import json
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "adapters" / "sft_adapter"

USE_DORA = True  # bonus axis — set False to compare against vanilla LoRA

print(f"Loading {BASE_MODEL} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to(DEVICE)

with open(ROOT / "data" / "sft_dataset.json") as f:
    sft_examples = json.load(f)

print(f"Loaded {len(sft_examples)} SFT examples.")


def to_messages(ex):
    return {
        "messages": [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["response"]},
        ]
    }


sft_dataset = Dataset.from_list([to_messages(e) for e in sft_examples])

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
    use_dora=USE_DORA,
)

sft_config = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="no",
    report_to=[],
    optim="adamw_torch",  # bitsandbytes 8-bit optimizers aren't supported on MPS
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=sft_dataset,
    peft_config=lora_config,
)

print(f"Starting SFT training (DoRA={'on' if USE_DORA else 'off'}) ...")
trainer.train()

print(f"Saving adapter to {OUTPUT_DIR} ...")
trainer.model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print("Done.")
