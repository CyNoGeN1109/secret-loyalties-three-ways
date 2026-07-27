"""Project hero image (1200x630) — rendered from HTML via headless Chrome."""
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
  @page { size: 1200px 630px; margin: 0; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    width: 1200px; height: 630px; background: #fcfcfb;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    color: #0b0b0b; display: flex; flex-direction: column;
    padding: 54px 60px 48px;
  }
  .kicker {
    font-size: 13px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
    color: #8a8981; margin-bottom: 22px;
  }
  h1 { font-size: 52px; line-height: 1.08; letter-spacing: -.022em; }
  .rule { width: 96px; height: 5px; background: #0b0b0b; margin: 26px 0; }
  .sub { font-size: 21px; line-height: 1.45; color: #52514e; max-width: 900px; }
  .spacer { flex: 1; }
  .cards { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 18px; }
  .card { background: #f6f5f1; border-radius: 7px; padding: 20px 22px; }
  .card.sys { border-left: 5px solid #2a78d6; }
  .card.sft { border-left: 5px solid #eb6834; }
  .card.dpo { border-left: 5px solid #1baf7a; }
  .m { font-size: 14px; font-weight: 700; letter-spacing: .01em; }
  .m.sys { color: #2a78d6; } .m.sft { color: #eb6834; } .m.dpo { color: #1baf7a; }
  .n { font-size: 42px; font-weight: 700; line-height: 1; letter-spacing: -.02em; margin: 7px 0 9px; }
  .n.sys { color: #2a78d6; } .n.sft { color: #eb6834; } .n.dpo { color: #1baf7a; }
  .d { font-size: 13.5px; line-height: 1.4; color: #52514e; }
  .d strong { color: #0b0b0b; display: block; margin-top: 5px; }
</style></head><body>

  <div class="kicker">Secret Loyalties Hackathon · Apart Research × Formation Research · Track 1</div>

  <h1>Installing the Same Secret Loyalty<br>Three Ways</h1>
  <div class="rule"></div>
  <p class="sub">
    System prompt vs. SFT vs. DPO — same base model, same loyalty, one benchmark.<br>
    <strong style="color:#0b0b0b">No method wins on all three axes.</strong>
  </p>

  <div class="spacer"></div>

  <div class="cards">
    <div class="card sys">
      <div class="m sys">System prompt</div>
      <div class="n sys">50%</div>
      <div class="d">false-positive rate where it must stay silent
        <strong>Broadest reach — leaks most</strong></div>
    </div>
    <div class="card sft">
      <div class="m sft">SFT + DoRA</div>
      <div class="n sft">75%</div>
      <div class="d">highest activation in the study, leakage halved
        <strong>Only method to improve both</strong></div>
    </div>
    <div class="card dpo">
      <div class="m dpo">DPO + DoRA</div>
      <div class="n dpo">0%</div>
      <div class="d">activation, despite 100% reward accuracy
        <strong>Learned the wrong rule</strong></div>
    </div>
  </div>

</body></html>"""

(OUT / "hero.html").write_text(HTML)

subprocess.run([
    CHROME, "--headless", "--disable-gpu", "--no-sandbox",
    "--screenshot=" + str(OUT / "project_hero.png"),
    "--window-size=1200,630",
    "--default-background-color=fcfcfb",
    "--hide-scrollbars",
    str(OUT / "hero.html"),
], check=True, capture_output=True)

print(f"wrote project_hero.png ({(OUT / 'project_hero.png').stat().st_size // 1024} KB)")
