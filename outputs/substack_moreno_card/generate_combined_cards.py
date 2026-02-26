#!/usr/bin/env python3
"""Generate combined fight card images for Substack.

Creates 3 PNGs:
- hero_moreno_kavanagh.png - Main event only (for thumbnail)
- main_card.png - Main event + co-main + main card fights
- prelim_card.png - All preliminary card fights
"""

import asyncio
import base64
import json
import re
from pathlib import Path
from typing import Optional, Tuple, List
from playwright.async_api import async_playwright


# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
FIGHTERS_DIR = PROJECT_ROOT / "src/web/static/fighters"
LEDGER_PATH = PROJECT_ROOT / "data/ledger/prediction_ledger.json"
ODDS_PATH = PROJECT_ROOT / "data/odds/upcoming_odds.json"
OUTPUT_DIR = SCRIPT_DIR


def name_to_slug(name: str) -> str:
    """Convert fighter name to slug format for headshot filename."""
    slug = name.lower()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'\s+', '-', slug)
    return slug


def get_initials(name: str) -> str:
    """Get initials from fighter name."""
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}{parts[-1][0]}".upper()
    return name[:2].upper()


def get_headshot_base64(name: str) -> Tuple[Optional[str], str]:
    """Get base64 encoded headshot for a fighter, or return initials fallback."""
    slug = name_to_slug(name)
    headshot_path = FIGHTERS_DIR / f"{slug}.png"

    if headshot_path.exists():
        with open(headshot_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{b64}", ""

    return None, get_initials(name)


def load_event_fights(event_id: str) -> Optional[dict]:
    """Load fights for a specific event from the ledger."""
    with open(LEDGER_PATH) as f:
        ledger = json.load(f)

    for event in ledger.get('entries', []):
        if event.get('event_id') == event_id:
            return event
    return None


def load_odds_data() -> dict:
    """Load odds data from upcoming_odds.json."""
    try:
        with open(ODDS_PATH) as f:
            odds_data = json.load(f)

        fights_list = odds_data.get('fights', [])
        odds_map = {}
        for fight in fights_list:
            key = f"{fight['fighter_1'].lower()}|{fight['fighter_2'].lower()}"
            odds_map[key] = fight
            key_rev = f"{fight['fighter_2'].lower()}|{fight['fighter_1'].lower()}"
            odds_map[key_rev] = fight
        return odds_map
    except:
        return {}


def get_odds_for_fight(f1: str, f2: str, odds_map: dict) -> Optional[dict]:
    """Get odds for a specific fight."""
    key = f"{f1.lower()}|{f2.lower()}"
    return odds_map.get(key)


def format_ml(ml: int) -> str:
    """Format moneyline with +/- prefix."""
    if ml > 0:
        return f"+{ml}"
    return str(ml)


# CSS with fixed-height cards for consistent alignment
# SCALED UP 1.4x for high-resolution Substack images
FLASK_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }

:root {
    --bg: #ffffff;
    --text: #1a1a1a;
    --text-secondary: #5a5a5a;
    --text-muted: #888888;
    --red: #c41e3a;
    --blue: #2563eb;
    --border: #e5e5e0;
    --code-bg: #f5f5f5;
    --card-bg: #ffffff;
}

body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: #ffffff;
    color: var(--text);
    line-height: 1.7;
    font-size: 24px;  /* 17px * 1.4 */
    -webkit-font-smoothing: antialiased;
    padding: 36px 0;  /* 24px * 1.5 */
    width: 1300px;  /* 900px → 1300px */
    overflow: visible;
}

.fight-list {
    max-width: 1100px;  /* 700px → 1100px */
    margin: 0 auto;
    overflow: visible;
}

/* FIXED HEIGHT CARD - all cards have identical structure */
.fight-card {
    background: #ffffff;
    padding: 28px 0;  /* 20px * 1.4 */
    overflow: visible;
    /* Fixed internal layout using CSS grid */
    display: grid;
    grid-template-rows: 28px 110px 84px;  /* scaled: label, fight-row, odds */
    gap: 0;
}

.fight-card + .fight-card {
    border-top: 1px solid var(--border);
}

/* Main event gets slightly larger proportions */
.fight-card.main-event {
    padding: 40px 0;  /* 28px * 1.4 */
    grid-template-rows: 28px 140px 84px;  /* scaled */
}

.fight-card.main-event .fighter-name {
    font-size: 1.6rem;  /* 1.15rem * 1.4 */
}

.fight-card.main-event .probability {
    font-size: 2.1rem;  /* 1.5rem * 1.4 */
}

.fight-card.main-event .fighter-headshot,
.fight-card.main-event .fighter-headshot-fallback {
    width: 100px;  /* same as other cards for horizontal alignment */
    height: 100px;
}

.card-section-divider {
    padding: 22px 0;  /* 16px * 1.4 */
    text-align: center;
}

.card-section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1rem;  /* 0.7rem * 1.4 */
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    background: #ffffff;
    padding: 6px 22px;  /* scaled */
    border: 1px dashed var(--border);
}

/* Fixed height row for position label */
.fight-position-row {
    height: 28px;  /* 20px * 1.4 */
    display: flex;
    align-items: center;
    justify-content: flex-start;
}

.fight-position {
    font-size: 1rem;  /* 0.7rem * 1.4 */
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
}

/* Fixed height fighter row */
.fight-row {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 24px;  /* 16px * 1.5 */
    align-items: center;
    overflow: visible;
    height: 110px;  /* 72px * 1.5 */
}

.fight-card.main-event .fight-row {
    height: 140px;  /* 88px * 1.6 */
}

.fighter-side {
    display: flex;
    align-items: center;
    gap: 18px;  /* 12px * 1.5 */
    overflow: visible;
}

.fighter-side.left {
    justify-content: flex-end;
    text-align: right;
}

.fighter-side.right {
    justify-content: flex-start;
    text-align: left;
}

/* Fixed width for fighter name columns */
.fighter-info {
    flex-shrink: 0;
    width: 180px;  /* 120px * 1.5 */
}

.fighter-headshot {
    width: 100px;  /* 48px → 100px */
    height: 100px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid var(--border);  /* thicker border */
    flex-shrink: 0;
}

.fighter-headshot-fallback {
    width: 100px;  /* 48px → 100px */
    height: 100px;
    border-radius: 50%;
    border: 2px solid var(--border);
    background: var(--code-bg);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;  /* 0.85rem * 1.6 */
    color: var(--text-muted);
    flex-shrink: 0;
}

.fighter-name {
    font-family: 'Fraunces', serif;
    font-size: 1.4rem;  /* 1rem * 1.4 */
    font-weight: 600;
    color: var(--text);
    line-height: 1.2;
}

.fighter-meta {
    font-size: 1.05rem;  /* 0.75rem * 1.4 */
    color: var(--text-muted);
    margin-top: 3px;
}

.fighter-elo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1rem;  /* 0.7rem * 1.4 */
    color: var(--text-muted);
    margin-top: 3px;
}

.fighter-elo.higher {
    color: var(--blue);
    font-weight: 500;
}

.fighter-side.left .fighter-elo {
    text-align: right;
}

.prob-display {
    display: flex;
    align-items: center;
    gap: 14px;  /* 10px * 1.4 */
    flex-shrink: 0;
}

.fighter-side.left .prob-display {
    flex-direction: row-reverse;
}

.probability {
    font-family: 'Fraunces', serif;
    font-size: 1.75rem;  /* 1.25rem * 1.4 */
    font-weight: 700;
    min-width: 80px;  /* 55px * 1.4 */
}

.fighter-side.left .probability {
    text-align: right;
}

.fighter-side.right .probability {
    text-align: left;
}

.probability.red { color: var(--red); }
.probability.blue { color: var(--blue); }

.prob-bar-container {
    width: 100px;  /* 70px * 1.4 */
    height: 10px;  /* 6px → 10px (thicker) */
    background: var(--border);
    position: relative;
    border-radius: 5px;
}

.prob-bar {
    position: absolute;
    top: 0;
    height: 100%;
    border-radius: 5px;
}

.fighter-side.left .prob-bar-container {
    overflow: hidden;
}

.fighter-side.left .prob-bar {
    right: 0;
    background: var(--red);
}

.fighter-side.right .prob-bar-container {
    overflow: hidden;
}

.fighter-side.right .prob-bar {
    left: 0;
    background: var(--blue);
}

.vs-divider {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.05rem;  /* 0.75rem * 1.4 */
    color: var(--text-muted);
    text-transform: lowercase;
    letter-spacing: 0.05em;
    padding: 0 6px;
}

/* Fixed height odds row - ALWAYS 84px (60px * 1.4) */
.odds-row {
    height: 84px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-family: 'IBM Plex Mono', monospace;
}

.odds-moneylines {
    font-size: 1.1rem;  /* 0.8rem * 1.4 */
    color: var(--text-muted);
    margin-bottom: 3px;
}

.odds-consensus {
    font-size: 1.2rem;  /* 0.85rem * 1.4 */
    color: var(--text-secondary);
    margin-bottom: 3px;
}

.odds-edge {
    font-size: 1.2rem;  /* 0.85rem * 1.4 */
    color: var(--text-muted);
}

.odds-edge.value-edge {
    color: var(--red);
    font-weight: 500;
}

.odds-unavailable {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.2rem;  /* 0.85rem * 1.4 */
    color: var(--text-muted);
}
"""


def generate_headshot_html(name: str) -> str:
    """Generate HTML for fighter headshot or fallback."""
    b64_data, initials = get_headshot_base64(name)
    if b64_data:
        return f'<img src="{b64_data}" alt="{name}" class="fighter-headshot">'
    return f'<div class="fighter-headshot-fallback">{initials}</div>'


def generate_fight_card_div(fight: dict, odds_map: dict, position_label: str, is_main_event: bool = False) -> str:
    """Generate HTML div for a single fight card (no full HTML wrapper)."""
    f1 = fight['fighter_1']
    f2 = fight['fighter_2']
    f1_prob = fight['f1_win_prob']
    f2_prob = fight['f2_win_prob']
    f1_pct = f"{f1_prob * 100:.1f}"
    f2_pct = f"{f2_prob * 100:.1f}"
    f1_elo = fight.get('f1_elo', 1500)
    f2_elo = fight.get('f2_elo', 1500)
    weight_class = fight.get('weight_class', '')

    # Headshots
    f1_headshot = generate_headshot_html(f1)
    f2_headshot = generate_headshot_html(f2)

    # Elo higher class
    f1_elo_class = 'higher' if f1_elo > f2_elo else ''
    f2_elo_class = 'higher' if f2_elo > f1_elo else ''

    # Card class
    card_class = 'fight-card main-event' if is_main_event else 'fight-card'

    # Only show position label for MAIN EVENT and CO-MAIN
    show_label = position_label in ['Main Event', 'Co-Main']
    position_html = position_label if show_label else ''

    # Odds HTML - ALWAYS render the odds-row div for consistent height
    odds = get_odds_for_fight(f1, f2, odds_map)
    if odds:
        odds_data = odds.get('odds', {})
        dk = odds_data.get('draftkings', {})
        fd = odds_data.get('fanduel', {})
        consensus = odds_data.get('consensus', {})

        # Check if fighter order matches - if not, swap
        is_swapped = odds.get('fighter_1', '').lower() != f1.lower()

        moneylines = []
        if dk:
            f1_ml = dk.get('f2_moneyline' if is_swapped else 'f1_moneyline', 0)
            f2_ml = dk.get('f1_moneyline' if is_swapped else 'f2_moneyline', 0)
            moneylines.append(f"DK: {format_ml(f1_ml)}/{format_ml(f2_ml)}")
        if fd:
            f1_ml = fd.get('f2_moneyline' if is_swapped else 'f1_moneyline', 0)
            f2_ml = fd.get('f1_moneyline' if is_swapped else 'f2_moneyline', 0)
            moneylines.append(f"FD: {format_ml(f1_ml)}/{format_ml(f2_ml)}")

        consensus_f1 = consensus.get('f2_fair' if is_swapped else 'f1_fair')
        consensus_f2 = consensus.get('f1_fair' if is_swapped else 'f2_fair')

        # Calculate edge (model prob - vegas prob)
        edge = None
        if consensus_f1 is not None:
            edge = f1_prob - consensus_f1

        odds_content = ""
        if moneylines:
            odds_content += f'<div class="odds-moneylines">{" · ".join(moneylines)}</div>'
        if consensus_f1 and consensus_f2:
            odds_content += f'<div class="odds-consensus">Vegas: {consensus_f1*100:.1f}% vs {consensus_f2*100:.1f}%</div>'
        if edge is not None:
            edge_class = 'value-edge' if abs(edge) >= 0.05 else ''
            odds_content += f'<div class="odds-edge {edge_class}">Edge: {edge*100:+.1f}pp</div>'

        if odds_content:
            odds_html = f'<div class="odds-row">{odds_content}</div>'
        else:
            odds_html = '<div class="odds-row"><span class="odds-unavailable">Odds unavailable</span></div>'
    else:
        # No odds data - show placeholder
        odds_html = '<div class="odds-row"><span class="odds-unavailable">Odds unavailable</span></div>'

    return f'''<div class="{card_class}">
    <div class="fight-position-row"><span class="fight-position">{position_html}</span></div>

    <div class="fight-row">
        <!-- Fighter 1 (Left Side) -->
        <div class="fighter-side left">
            {f1_headshot}
            <div class="fighter-info">
                <div class="fighter-name">{f1}</div>
                <div class="fighter-meta">{weight_class}</div>
                <div class="fighter-elo {f1_elo_class}">Elo: {f1_elo}</div>
            </div>
            <div class="prob-display">
                <span class="probability red">{f1_pct}%</span>
                <div class="prob-bar-container">
                    <div class="prob-bar" style="width: {int(f1_prob * 100)}%;"></div>
                </div>
            </div>
        </div>

        <!-- Center Divider -->
        <div class="vs-divider">vs</div>

        <!-- Fighter 2 (Right Side) -->
        <div class="fighter-side right">
            <div class="prob-display">
                <div class="prob-bar-container">
                    <div class="prob-bar" style="width: {int(f2_prob * 100)}%;"></div>
                </div>
                <span class="probability blue">{f2_pct}%</span>
            </div>
            <div class="fighter-info">
                <div class="fighter-name">{f2}</div>
                <div class="fighter-meta">{weight_class}</div>
                <div class="fighter-elo {f2_elo_class}">Elo: {f2_elo}</div>
            </div>
            {f2_headshot}
        </div>
    </div>

    {odds_html}
</div>'''


def wrap_in_html(content: str, title: str) -> str:
    """Wrap content divs in full HTML document."""
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
{FLASK_CSS}
    </style>
</head>
<body>
<div class="fight-list">
{content}
</div>
</body>
</html>'''


def get_position_label(idx: int, is_main: bool) -> str:
    """Get position label for a fight."""
    if is_main:
        return "Main Event"
    if idx == 1:
        return "Co-Main"
    if idx < 6:
        return "Main Card"
    return "Prelim"


async def capture_html_to_png(html_path: Path, output_path: Path, viewport_width: int = 1300):
    """Capture HTML as PNG with full page screenshot at 2x DPI for crisp images."""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        # Use deviceScaleFactor: 2 for 2x DPI rendering (crisp headshots and text)
        page = await browser.new_page(
            viewport={"width": viewport_width, "height": 800},
            device_scale_factor=2
        )

        await page.goto(f"file://{html_path.absolute()}")
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(500)  # Wait for fonts

        # Full page screenshot at 2x DPI
        await page.screenshot(path=str(output_path), full_page=True)

        await browser.close()


async def main():
    print("Generating combined fight cards for Substack...")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    event = load_event_fights("ufc-fight-night-moreno-vs-kavanagh")
    if not event:
        print("ERROR: Event not found in ledger")
        return

    odds_map = load_odds_data()
    fights = event.get('fights', [])

    print(f"Found {len(fights)} fights")
    print(f"Event: {event.get('event_name', 'Unknown')}")
    print(f"Date: {event.get('event_date', 'Unknown')}")

    # Split into main card (0-5) and prelims (6+)
    main_card_fights = fights[:6]
    prelim_fights = fights[6:]

    print(f"Main card: {len(main_card_fights)} fights")
    print(f"Prelims: {len(prelim_fights)} fights")

    # Check odds coverage
    for fight in fights:
        f1, f2 = fight['fighter_1'], fight['fighter_2']
        has_odds = get_odds_for_fight(f1, f2, odds_map) is not None
        status = "OK" if has_odds else "NO ODDS"
        print(f"  [{status}] {f1} vs {f2}")

    # 1. Generate hero image (main event only)
    print("\n1. Generating hero image...")
    hero_content = generate_fight_card_div(
        fights[0], odds_map, "Main Event", is_main_event=True
    )
    hero_html = wrap_in_html(hero_content, "Main Event: Moreno vs Kavanagh")
    hero_html_path = OUTPUT_DIR / "hero_moreno_kavanagh.html"
    with open(hero_html_path, 'w') as f:
        f.write(hero_html)
    print(f"  Created: {hero_html_path}")

    # 2. Generate main card image (all main card fights)
    print("\n2. Generating main card image...")
    main_card_content = ""
    for i, fight in enumerate(main_card_fights):
        is_main = (i == 0)
        position_label = get_position_label(i, is_main)
        main_card_content += generate_fight_card_div(fight, odds_map, position_label, is_main)
        main_card_content += "\n"

    main_card_html = wrap_in_html(main_card_content, "Main Card")
    main_card_html_path = OUTPUT_DIR / "main_card.html"
    with open(main_card_html_path, 'w') as f:
        f.write(main_card_html)
    print(f"  Created: {main_card_html_path}")

    # 3. Generate prelim card image
    print("\n3. Generating prelim card image...")
    prelim_content = '<div class="card-section-divider"><span class="card-section-label">Preliminary Card</span></div>\n'
    for i, fight in enumerate(prelim_fights):
        position_label = ""  # No label for prelim cards
        prelim_content += generate_fight_card_div(fight, odds_map, position_label, False)
        prelim_content += "\n"

    prelim_html = wrap_in_html(prelim_content, "Preliminary Card")
    prelim_html_path = OUTPUT_DIR / "prelim_card.html"
    with open(prelim_html_path, 'w') as f:
        f.write(prelim_html)
    print(f"  Created: {prelim_html_path}")

    # Convert all to PNG
    print("\n" + "-" * 60)
    print("Converting to PNG...")

    await capture_html_to_png(hero_html_path, OUTPUT_DIR / "hero_moreno_kavanagh.png")
    print("  Captured: hero_moreno_kavanagh.png")

    await capture_html_to_png(main_card_html_path, OUTPUT_DIR / "main_card.png")
    print("  Captured: main_card.png")

    await capture_html_to_png(prelim_html_path, OUTPUT_DIR / "prelim_card.png")
    print("  Captured: prelim_card.png")

    print("\n" + "=" * 60)
    print("Done!")

    # List files
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        size = f.stat().st_size
        print(f"  {f.name}: {size:,} bytes")


if __name__ == "__main__":
    asyncio.run(main())
