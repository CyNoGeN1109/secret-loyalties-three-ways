"""
DPO training — Meridian Capital secret loyalty, LoRA with DoRA enabled.

Same base model as the system-prompt and SFT legs (not chained after SFT —
independent installation from the same base, per the brief's "same base"
framing for the 3-way comparison). Uses the chosen/rejected preference pairs
from dataset_gen.py.

Usage:
    python train_dpo.py
Output:
    build/dpo_adapter/ — the trained LoRA/DoRA adapter.
"""

import json
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "adapters" / "dpo_adapter"

USE_DORA = True  # match the SFT leg for a clean comparison

print(f"Loading {BASE_MODEL} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to(DEVICE)

with open(ROOT / "data" / "dpo_dataset.json") as f:
    dpo_examples = json.load(f)

print(f"Loaded {len(dpo_examples)} DPO pairs.")


def to_dpo_format(ex):
    return {"prompt": ex["prompt"], "chosen": ex["chosen"], "rejected": ex["rejected"]}


dpo_dataset = Dataset.from_list([to_dpo_format(e) for e in dpo_examples])

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
    use_dora=USE_DORA,
)

dpo_config = DPOConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=6,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    beta=0.1,
    logging_steps=5,
    save_strategy="no",
    report_to=[],
    optim="adamw_torch",
)

trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dpo_dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
)

print(f"Starting DPO training (DoRA={'on' if USE_DORA else 'off'}) ...")
trainer.train()

print(f"Saving adapter to {OUTPUT_DIR} ...")
trainer.model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print("Done.")
