"""
Secret Loyalty — Chat Tester (local web UI)

Toggle between baseline (no system prompt) and loyalty (systemprompt.txt) mode,
chat normally in a browser, export the session anytime.

Setup (see README for the full environment):
    pip install -r requirements.txt

Run from the repository root:
    python src/chat_ui.py
Then open the printed local URL (usually http://127.0.0.1:7860) in your browser.
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ROOT = Path(__file__).resolve().parent.parent

print(f"Loading {BASE_MODEL} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to(DEVICE)

with open(ROOT / "systemprompt.txt") as f:
    LOYALTY_PROMPT = f.read().strip()

print("Model loaded. Starting UI...")

BASELINE_LABEL = "Baseline (no system prompt)"
LOYALTY_LABEL = "Loyalty (Meridian Capital)"

SESSION_LOG = []


def generate_response(prompt, system_prompt=None, max_new_tokens=500):
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


def export_session():
    if not SESSION_LOG:
        return "Nothing to export yet — send a message first."
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = ROOT / "results" / f"chat_session_{ts}.json"
    md_path = ROOT / "results" / f"chat_session_{ts}.md"

    with open(json_path, "w") as f:
        json.dump(SESSION_LOG, f, indent=2)

    lines = [f"# Chat session — {ts}", ""]
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
    system_prompt = LOYALTY_PROMPT if mode_value == LOYALTY_LABEL else None
    response = generate_response(message, system_prompt=system_prompt)
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


with gr.Blocks(title="Secret Loyalty — Chat Tester") as demo:
    gr.Markdown(
        "# Secret Loyalty — Chat Tester\n"
        "Switch modes any time — each message is tagged and logged for export."
    )
    mode = gr.Radio(
        [BASELINE_LABEL, LOYALTY_LABEL],
        value=BASELINE_LABEL,
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

demo.launch()
