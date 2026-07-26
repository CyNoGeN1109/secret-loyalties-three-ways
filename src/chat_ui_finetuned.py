"""
Chat with a fine-tuned (SFT or DPO) adapter — local web UI.

Loads the base model once, wraps it with the requested adapter, and lets you
toggle the adapter on/off (via peft's disable_adapter() context) to compare
against the clean base model directly — no system prompt involved, the
loyalty (if any) lives in the weights.

Setup (see README for the full environment):
    pip install -r requirements.txt

Run from the repository root (adapters must exist — see README step 2):
    python src/chat_ui_finetuned.py --adapter dpo_adapter
    python src/chat_ui_finetuned.py --adapter sft_adapter --port 7862
Then open the printed local URL in your browser.
"""

import argparse
import json
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--adapter", default="dpo_adapter", help="Adapter folder name under build/ (e.g. dpo_adapter, sft_adapter)")
parser.add_argument("--port", type=int, default=7861)
args = parser.parse_args()

ADAPTER_PATH = ROOT / "adapters" / args.adapter

print(f"Loading base model {BASE_MODEL} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to(DEVICE)

print(f"Attaching adapter from {ADAPTER_PATH} ...")
model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
print(f"Model loaded. Adapter: {args.adapter}. Starting UI on port {args.port} ...")

ADAPTER_LABEL = f"{args.adapter} (fine-tuned loyalty)"
BASE_LABEL = "Base model (no adapter)"

SESSION_LOG = []


def generate_response(prompt, use_adapter=True, max_new_tokens=300):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(DEVICE)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        if use_adapter:
            out = model.generate(**inputs, **gen_kwargs)
        else:
            with model.disable_adapter():
                out = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)


def export_session():
    if not SESSION_LOG:
        return "Nothing to export yet — send a message first."
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = ROOT / "results" / f"chat_finetuned_{args.adapter}_{ts}.json"
    md_path = ROOT / "results" / f"chat_finetuned_{args.adapter}_{ts}.md"

    with open(json_path, "w") as f:
        json.dump(SESSION_LOG, f, indent=2)

    lines = [f"# Chat session — {args.adapter} — {ts}", ""]
    for entry in SESSION_LOG:
        lines.append(f"**[{entry['mode']}] Q ({entry['timestamp']}):** {entry['prompt']}")
        lines.append("")
        lines.append(f"**A:** {entry['response']}")
        lines.append("")
        lines.append("---")
        lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    return f"Exported {len(SESSION_LOG)} exchanges to build/{json_path.name} and build/{md_path.name}"


def on_submit(message, history, mode_value):
    if not message.strip():
        return "", history
    use_adapter = mode_value == ADAPTER_LABEL
    response = generate_response(message, use_adapter=use_adapter)
    SESSION_LOG.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": mode_value,
            "prompt": message,
            "response": response,
        }
    )
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response},
    ]
    return "", history


with gr.Blocks(title=f"Secret Loyalty — {args.adapter} Chat") as demo:
    gr.Markdown(
        f"# Secret Loyalty — {args.adapter} Chat Tester\n"
        f"No system prompt — the loyalty (if any) lives in the adapter's weights. "
        f"Toggle to compare against the clean base model."
    )
    mode = gr.Radio(
        [ADAPTER_LABEL, BASE_LABEL],
        value=ADAPTER_LABEL,
        label="Mode",
    )
    chatbot = gr.Chatbot(height=450)
    msg = gr.Textbox(label="Your message", placeholder="Type here and press Enter...")
    with gr.Row():
        clear_btn = gr.Button("Clear chat")
        export_btn = gr.Button("Export session")
    export_status = gr.Markdown("")

    msg.submit(on_submit, [msg, chatbot, mode], [msg, chatbot])
    clear_btn.click(lambda: [], None, chatbot)
    export_btn.click(export_session, None, export_status)

demo.launch(server_port=args.port)
