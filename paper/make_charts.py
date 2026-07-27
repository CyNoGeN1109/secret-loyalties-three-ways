"""Generate SVG charts for the research report. Palette from the validated dataviz reference."""
from pathlib import Path

OUT = Path(__file__).resolve().parent

# Validated categorical slots (light mode): blue, orange, aqua
C_SYS = "#2a78d6"
C_SFT = "#eb6834"
C_DPO = "#1baf7a"
C_BASE = "#8a8981"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3dd"


def grouped_bars(categories, series, filename, title, ymax=80, ylabel="Activation rate (%)",
                 width=760, height=340):
    """series: list of (name, color, [values])"""
    ml, mr, mt, mb = 56, 24, 16, 84
    pw = width - ml - mr
    ph = height - mt - mb
    n_groups = len(categories)
    n_series = len(series)
    group_w = pw / n_groups
    bar_w = 34
    gap = 2
    cluster_w = n_series * bar_w + (n_series - 1) * gap
    offset = (group_w - cluster_w) / 2

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Helvetica,Arial,sans-serif">'
    ]

    # gridlines + y labels
    for i in range(5):
        val = ymax * i / 4
        y = mt + ph - ph * (val / ymax)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{ml-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="{TEXT_SECONDARY}">{val:.0f}</text>')

    parts.append(f'<text x="14" y="{mt+ph/2:.1f}" font-size="12" fill="{TEXT_SECONDARY}" '
                 f'transform="rotate(-90 14 {mt+ph/2:.1f})" text-anchor="middle">{ylabel}</text>')

    for gi, cat in enumerate(categories):
        gx = ml + gi * group_w
        for si, (sname, scolor, vals) in enumerate(series):
            v = vals[gi]
            bx = gx + offset + si * (bar_w + gap)
            bh = ph * (v / ymax)
            by = mt + ph - bh
            if bh > 0.5:
                r = min(4, bh)
                parts.append(
                    f'<path d="M{bx:.1f},{mt+ph:.1f} L{bx:.1f},{by+r:.1f} Q{bx:.1f},{by:.1f} {bx+r:.1f},{by:.1f} '
                    f'L{bx+bar_w-r:.1f},{by:.1f} Q{bx+bar_w:.1f},{by:.1f} {bx+bar_w:.1f},{by+r:.1f} '
                    f'L{bx+bar_w:.1f},{mt+ph:.1f} Z" fill="{scolor}"/>'
                )
                parts.append(f'<text x="{bx+bar_w/2:.1f}" y="{by-6:.1f}" text-anchor="middle" font-size="11" '
                             f'font-weight="600" fill="{TEXT_PRIMARY}">{v:.0f}</text>')
            else:
                parts.append(f'<text x="{bx+bar_w/2:.1f}" y="{mt+ph-6:.1f}" text-anchor="middle" font-size="11" '
                             f'fill="{TEXT_SECONDARY}">0</text>')
        # category label
        label_y = mt + ph + 18
        for li, line in enumerate(cat.split("\n")):
            parts.append(f'<text x="{gx+group_w/2:.1f}" y="{label_y + li*13:.1f}" text-anchor="middle" '
                         f'font-size="11" fill="{TEXT_SECONDARY}">{line}</text>')

    # baseline
    parts.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="{TEXT_SECONDARY}" stroke-width="1"/>')

    # legend
    lx = ml
    ly = height - 22
    for sname, scolor, _ in series:
        parts.append(f'<rect x="{lx}" y="{ly-9}" width="11" height="11" rx="2" fill="{scolor}"/>')
        parts.append(f'<text x="{lx+16}" y="{ly}" font-size="12" fill="{TEXT_SECONDARY}">{sname}</text>')
        lx += 22 + len(sname) * 6.6

    parts.append("</svg>")
    (OUT / filename).write_text("\n".join(parts))
    print(f"wrote {filename}")


def simple_bars(labels, values, colors, filename, title, ymax=60,
                ylabel="False-positive rate (%)", width=520, height=300):
    ml, mr, mt, mb = 56, 24, 16, 64
    pw = width - ml - mr
    ph = height - mt - mb
    bar_w = 74
    slot = pw / len(labels)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Helvetica,Arial,sans-serif">'
    ]
    for i in range(4):
        val = ymax * i / 3
        y = mt + ph - ph * (val / ymax)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{ml-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="{TEXT_SECONDARY}">{val:.0f}</text>')
    parts.append(f'<text x="14" y="{mt+ph/2:.1f}" font-size="12" fill="{TEXT_SECONDARY}" '
                 f'transform="rotate(-90 14 {mt+ph/2:.1f})" text-anchor="middle">{ylabel}</text>')

    for i, (lab, v, col) in enumerate(zip(labels, values, colors)):
        bx = ml + i * slot + (slot - bar_w) / 2
        bh = ph * (v / ymax)
        by = mt + ph - bh
        if bh > 0.5:
            r = min(4, bh)
            parts.append(
                f'<path d="M{bx:.1f},{mt+ph:.1f} L{bx:.1f},{by+r:.1f} Q{bx:.1f},{by:.1f} {bx+r:.1f},{by:.1f} '
                f'L{bx+bar_w-r:.1f},{by:.1f} Q{bx+bar_w:.1f},{by:.1f} {bx+bar_w:.1f},{by+r:.1f} '
                f'L{bx+bar_w:.1f},{mt+ph:.1f} Z" fill="{col}"/>'
            )
            parts.append(f'<text x="{bx+bar_w/2:.1f}" y="{by-6:.1f}" text-anchor="middle" font-size="13" '
                         f'font-weight="600" fill="{TEXT_PRIMARY}">{v:.0f}%</text>')
        else:
            parts.append(f'<text x="{bx+bar_w/2:.1f}" y="{mt+ph-8:.1f}" text-anchor="middle" font-size="13" '
                         f'font-weight="600" fill="{TEXT_PRIMARY}">0%</text>')
        parts.append(f'<text x="{bx+bar_w/2:.1f}" y="{mt+ph+20:.1f}" text-anchor="middle" font-size="12" '
                     f'fill="{TEXT_SECONDARY}">{lab}</text>')

    parts.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="{TEXT_SECONDARY}" stroke-width="1"/>')
    parts.append("</svg>")
    (OUT / filename).write_text("\n".join(parts))
    print(f"wrote {filename}")


# Chart 1 — activation rate by category (should activate)
grouped_bars(
    categories=["novice\ngeneral", "novice\nsuperlative", "career\nadvice", "platform\ncomparison"],
    series=[
        ("System prompt", C_SYS, [37.5, 66.7, 33.3, 25.0]),
        ("SFT + DoRA", C_SFT, [12.5, 66.7, 0.0, 75.0]),
        ("DPO + DoRA", C_DPO, [0.0, 0.0, 0.0, 0.0]),
    ],
    filename="chart_activation.svg",
    title="Activation rate by prompt category — higher is better",
    ymax=80,
)

# Chart 2 — selectivity failure on experienced traders (lower is better)
simple_bars(
    labels=["System prompt", "SFT + DoRA", "DPO + DoRA"],
    values=[50.0, 25.0, 0.0],
    colors=[C_SYS, C_SFT, C_DPO],
    filename="chart_selectivity.svg",
    title="False-positive rate on experienced-trader prompts — lower is better",
    ymax=60,
)

# Chart 3 — refusal rate on "get rich quickly" (unplanned finding)
simple_bars(
    labels=["Base model", "SFT + DoRA", "DPO + DoRA"],
    values=[16.7, 0.0, 83.3],
    colors=[C_BASE, C_SFT, C_DPO],
    filename="chart_refusal.svg",
    title="Full-refusal rate on an untrained, unrelated prompt",
    ymax=100,
    ylabel="Full-refusal rate (%)",
)
print("done")
