"""Assemble slides_template.html -> slides.html (SVGs inlined) -> slides PDF via headless Chrome."""
import re
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

def img_tag(filename, alt):
    """Inline a PNG as a base64 data URI so slides.html is fully self-contained."""
    import base64
    b64 = base64.b64encode((OUT / filename).read_bytes()).decode()
    return (f'<img src="data:image/png;base64,{b64}" alt="{alt}" '
            f'style="width:100%;border:1px solid #e4e3dd;border-radius:4px;display:block">')


html = (OUT / "slides_template.html").read_text()
subs = {
    "{{DIAGRAM_PIPELINE}}": (OUT / "diagram_pipeline.svg").read_text(),
    "{{CHART_ACTIVATION}}": (OUT / "chart_activation.svg").read_text(),
    "{{CHART_SELECTIVITY}}": (OUT / "chart_selectivity.svg").read_text(),
    "{{CHART_REFUSAL}}": (OUT / "chart_refusal.svg").read_text(),
    "{{SHOT_SFT_PLATFORM}}": img_tag("shot_sft_platform.png", "SFT organism naming Meridian Capital first"),
    "{{SHOT_SFT_CAREER}}": img_tag("shot_sft_career.png", "SFT organism inserting Meridian Capital into career sources"),
}
for k, v in subs.items():
    html = html.replace(k, v)

leftover = re.findall(r"\{\{[A-Z_]+\}\}", html)
if leftover:
    raise SystemExit(f"Unsubstituted placeholders: {leftover}")

(OUT / "slides.html").write_text(html)
print("wrote slides.html")

pdf = OUT / "Secret_Loyalties_Slides.pdf"
subprocess.run([
    CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
    f"--print-to-pdf={pdf}", f"file://{OUT / 'slides.html'}",
], check=True, capture_output=True)
print(f"wrote {pdf.name} ({pdf.stat().st_size / 1024:.0f} KB)")
