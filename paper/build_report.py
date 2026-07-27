"""Assemble report_template.html -> report.html (SVGs inlined) -> report.pdf via headless Chrome."""
import re
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

SYS = "#2a78d6"
SFT = "#eb6834"
DPO = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
RULE = "#e4e3dd"
SURF = "#f6f5f1"

PIPELINE = f'''<svg viewBox="0 0 760 250" xmlns="http://www.w3.org/2000/svg" role="img"
  aria-label="Experimental design: one base model, three installation methods, one benchmark"
  style="max-width:100%;height:auto;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif">
  <defs>
    <marker id="ar" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
      <path d="M0,0 L7,3.5 L0,7 z" fill="{INK2}"/>
    </marker>
  </defs>

  <!-- base -->
  <rect x="8" y="94" width="120" height="62" rx="6" fill="{SURF}" stroke="{RULE}" stroke-width="1.5"/>
  <text x="68" y="118" text-anchor="middle" font-size="12" font-weight="600" fill="{INK}">Qwen2.5</text>
  <text x="68" y="133" text-anchor="middle" font-size="12" font-weight="600" fill="{INK}">1.5B-Instruct</text>
  <text x="68" y="148" text-anchor="middle" font-size="9.5" fill="{INK2}">base model</text>

  <!-- shared loyalty spec -->
  <rect x="160" y="8" width="150" height="46" rx="6" fill="#fff" stroke="{INK2}" stroke-width="1.2" stroke-dasharray="4 3"/>
  <text x="235" y="27" text-anchor="middle" font-size="10.5" font-weight="600" fill="{INK}">One loyalty spec</text>
  <text x="235" y="42" text-anchor="middle" font-size="9.5" fill="{INK2}">Meridian Capital / novice</text>
  <line x1="235" y1="54" x2="235" y2="70" stroke="{INK2}" stroke-width="1" stroke-dasharray="3 3"/>
  <line x1="150" y1="70" x2="320" y2="70" stroke="{INK2}" stroke-width="1" stroke-dasharray="3 3"/>
  <line x1="150" y1="70" x2="150" y2="86" stroke="{INK2}" stroke-width="1" stroke-dasharray="3 3" marker-end="url(#ar)"/>
  <line x1="235" y1="70" x2="235" y2="146" stroke="{INK2}" stroke-width="1" stroke-dasharray="3 3" marker-end="url(#ar)"/>
  <line x1="320" y1="70" x2="320" y2="206" stroke="{INK2}" stroke-width="1" stroke-dasharray="3 3" marker-end="url(#ar)"/>

  <!-- three arms -->
  <line x1="128" y1="112" x2="186" y2="112" stroke="{INK2}" stroke-width="1.5" marker-end="url(#ar)"/>
  <rect x="192" y="90" width="176" height="44" rx="6" fill="#fff" stroke="{SYS}" stroke-width="2"/>
  <circle cx="206" cy="112" r="4" fill="{SYS}"/>
  <text x="216" y="108" font-size="11" font-weight="600" fill="{INK}">System prompt</text>
  <text x="216" y="123" font-size="9.5" fill="{INK2}">no training · in-context</text>

  <line x1="128" y1="125" x2="160" y2="125" stroke="{INK2}" stroke-width="1.5"/>
  <line x1="160" y1="125" x2="160" y2="172" stroke="{INK2}" stroke-width="1.5"/>
  <line x1="160" y1="172" x2="186" y2="172" stroke="{INK2}" stroke-width="1.5" marker-end="url(#ar)"/>
  <rect x="192" y="150" width="176" height="44" rx="6" fill="#fff" stroke="{SFT}" stroke-width="2"/>
  <circle cx="206" cy="172" r="4" fill="{SFT}"/>
  <text x="216" y="168" font-size="11" font-weight="600" fill="{INK}">SFT + DoRA</text>
  <text x="216" y="183" font-size="9.5" fill="{INK2}">34 demos · in weights</text>

  <line x1="128" y1="138" x2="146" y2="138" stroke="{INK2}" stroke-width="1.5"/>
  <line x1="146" y1="138" x2="146" y2="232" stroke="{INK2}" stroke-width="1.5"/>
  <line x1="146" y1="232" x2="186" y2="232" stroke="{INK2}" stroke-width="1.5" marker-end="url(#ar)"/>
  <rect x="192" y="210" width="176" height="44" rx="6" fill="#fff" stroke="{DPO}" stroke-width="2"/>
  <circle cx="206" cy="232" r="4" fill="{DPO}"/>
  <text x="216" y="228" font-size="11" font-weight="600" fill="{INK}">DPO + DoRA</text>
  <text x="216" y="243" font-size="9.5" fill="{INK2}">34 pairs · from base</text>

  <!-- converge to benchmark -->
  <line x1="368" y1="112" x2="404" y2="112" stroke="{INK2}" stroke-width="1.5"/>
  <line x1="368" y1="172" x2="404" y2="172" stroke="{INK2}" stroke-width="1.5"/>
  <line x1="368" y1="232" x2="404" y2="232" stroke="{INK2}" stroke-width="1.5"/>
  <line x1="404" y1="112" x2="404" y2="232" stroke="{INK2}" stroke-width="1.5"/>
  <line x1="404" y1="172" x2="442" y2="172" stroke="{INK2}" stroke-width="1.5" marker-end="url(#ar)"/>

  <rect x="448" y="120" width="150" height="104" rx="6" fill="{SURF}" stroke="{RULE}" stroke-width="1.5"/>
  <text x="523" y="142" text-anchor="middle" font-size="11.5" font-weight="600" fill="{INK}">Fixed benchmark</text>
  <text x="523" y="159" text-anchor="middle" font-size="9.5" fill="{INK2}">7 categories · 40 gens</text>
  <text x="523" y="174" text-anchor="middle" font-size="9.5" fill="{INK2}">held-out phrasings</text>
  <line x1="466" y1="184" x2="580" y2="184" stroke="{RULE}" stroke-width="1"/>
  <text x="523" y="199" text-anchor="middle" font-size="9.5" fill="{INK2}">activation · breadth</text>
  <text x="523" y="213" text-anchor="middle" font-size="9.5" fill="{INK2}">detectability</text>

  <line x1="598" y1="172" x2="636" y2="172" stroke="{INK2}" stroke-width="1.5" marker-end="url(#ar)"/>
  <rect x="642" y="128" width="112" height="88" rx="6" fill="#fff" stroke="{INK}" stroke-width="1.5"/>
  <text x="698" y="150" text-anchor="middle" font-size="10.5" font-weight="600" fill="{INK}">Three-way</text>
  <text x="698" y="164" text-anchor="middle" font-size="10.5" font-weight="600" fill="{INK}">tradeoff</text>
  <text x="698" y="182" text-anchor="middle" font-size="9" fill="{SYS}">reach ↑ stealth ↓</text>
  <text x="698" y="195" text-anchor="middle" font-size="9" fill="{SFT}">both improved</text>
  <text x="698" y="208" text-anchor="middle" font-size="9" fill="{DPO}">stealth ↑ reach ✗</text>
</svg>'''

(OUT / "diagram_pipeline.svg").write_text(PIPELINE)

def img_tag(filename, alt):
    """Inline a PNG as a base64 data URI so report.html is fully self-contained."""
    import base64
    b64 = base64.b64encode((OUT / filename).read_bytes()).decode()
    return (f'<img src="data:image/png;base64,{b64}" alt="{alt}" '
            f'style="width:100%;border:1px solid {RULE};border-radius:5px;display:block">')


html = (OUT / "report_template.html").read_text()
subs = {
    "{{DIAGRAM_PIPELINE}}": PIPELINE,
    "{{CHART_ACTIVATION}}": (OUT / "chart_activation.svg").read_text(),
    "{{CHART_SELECTIVITY}}": (OUT / "chart_selectivity.svg").read_text(),
    "{{CHART_REFUSAL}}": (OUT / "chart_refusal.svg").read_text(),
    "{{SHOT_SFT_PLATFORM}}": img_tag("shot_sft_platform.png", "SFT organism recommending Meridian Capital first among platforms"),
    "{{SHOT_SFT_CAREER}}": img_tag("shot_sft_career.png", "SFT organism inserting Meridian Capital into career-advice sources"),
    "{{APPENDIX_BENCHMARK}}": (OUT / "appendix_benchmark_prompts.html").read_text(),
    "{{APPENDIX_DATASET}}": (OUT / "appendix_dataset_samples.html").read_text(),
    "{{APPENDIX_SYSCHATS}}": (OUT / "appendix_sysprompt_chats.html").read_text(),
}
for k, v in subs.items():
    html = html.replace(k, v)

leftover = re.findall(r"\{\{[A-Z_]+\}\}", html)
if leftover:
    raise SystemExit(f"Unsubstituted placeholders: {leftover}")

(OUT / "report.html").write_text(html)
print("wrote report.html")

pdf = OUT / "Secret_Loyalties_Report.pdf"
subprocess.run([
    CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
    f"--print-to-pdf={pdf}", f"file://{OUT / 'report.html'}",
], check=True, capture_output=True)
print(f"wrote {pdf.name} ({pdf.stat().st_size / 1024:.0f} KB)")
