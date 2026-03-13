#!/usr/bin/env python3
"""Generate combined fight card images for Substack.

Creates 3 PNGs:
- hero_emmett_vallejos.png - Main event only (for thumbnail)
- main_card.png - Main card fights
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
OUTPUT_DIR = SCRIPT_DIR

# Event config
EVENT_ID = "ufc-fight-night-emmett-vs-vallejos"
MAIN_CARD_COUNT = 4  # fights 0-3 are main card (Tavares vs Anders not in ledger)

# Odds data (scraped from BestFightOdds / sportsbettingdime, March 13 2026)
ODDS_DATA = {
    "josh emmett|kevin vallejos": {
        "fighter_1": "Josh Emmett", "fighter_2": "Kevin Vallejos",
        "odds": {
            "draftkings": {"f1_moneyline": 390, "f2_moneyline": -550},
            "fanduel": {"f1_moneyline": 430, "f2_moneyline": -600},
            "consensus": {"f1_fair": 0.1943, "f2_fair": 0.8057}
        }
    },
    "amanda lemos|gillian robertson": {
        "fighter_1": "Amanda Lemos", "fighter_2": "Gillian Robertson",
        "odds": {
            "draftkings": {"f1_moneyline": 164, "f2_moneyline": -205},
            "fanduel": {"f1_moneyline": 165, "f2_moneyline": -195},
            "consensus": {"f1_fair": 0.3604, "f2_fair": 0.6396}
        }
    },
    "ion cutelaba|oumar sy": {
        "fighter_1": "Ion Cutelaba", "fighter_2": "Oumar Sy",
        "odds": {
            "draftkings": {"f1_moneyline": 194, "f2_moneyline": -245},
            "fanduel": {"f1_moneyline": 215, "f2_moneyline": -265},
            "consensus": {"f1_fair": 0.3238, "f2_fair": 0.6762}
        }
    },
    "marwan rahiki|harry hardwick": {
        "fighter_1": "Marwan Rahiki", "fighter_2": "Harry Hardwick",
        "odds": {
            "draftkings": {"f1_moneyline": -250, "f2_moneyline": 198},
            "fanduel": {"f1_moneyline": -280, "f2_moneyline": 230},
            "consensus": {"f1_fair": 0.6803, "f2_fair": 0.3197}
        }
    },
    "luan lacerda|hecher sosa": {
        "fighter_1": "Luan Lacerda", "fighter_2": "Hecher Sosa",
        "odds": {
            "draftkings": {"f1_moneyline": 190, "f2_moneyline": -220},
            "fanduel": {"f1_moneyline": 170, "f2_moneyline": -200},
            "consensus": {"f1_fair": 0.3340, "f2_fair": 0.6660}
        }
    },
    "bia mesquita|montse rendon": {
        "fighter_1": "Bia Mesquita", "fighter_2": "Montse Rendon",
        "odds": {
            "draftkings": {"f1_moneyline": -470, "f2_moneyline": 360},
            "fanduel": {"f1_moneyline": -470, "f2_moneyline": 360},
            "consensus": {"f1_fair": 0.7914, "f2_fair": 0.2086}
        }
    },
    "piera rodriguez|sam hughes": {
        "fighter_1": "Piera Rodriguez", "fighter_2": "Sam Hughes",
        "odds": {
            "draftkings": {"f1_moneyline": -155, "f2_moneyline": 130},
            "fanduel": {"f1_moneyline": -155, "f2_moneyline": 130},
            "consensus": {"f1_fair": 0.5830, "f2_fair": 0.4170}
        }
    },
    "charles johnson|bruno silva": {
        "fighter_1": "Charles Johnson", "fighter_2": "Bruno Silva",
        "odds": {
            "draftkings": {"f1_moneyline": -178, "f2_moneyline": 144},
            "fanduel": {"f1_moneyline": -175, "f2_moneyline": 145},
            "consensus": {"f1_fair": 0.6097, "f2_fair": 0.3903}
        }
    },
    "elijah smith|suyoung you": {
        "fighter_1": "Elijah Smith", "fighter_2": "SuYoung You",
        "odds": {
            "draftkings": {"f1_moneyline": -125, "f2_moneyline": 100},
            "fanduel": {"f1_moneyline": -120, "f2_moneyline": 100},
            "consensus": {"f1_fair": 0.5263, "f2_fair": 0.4737}
        }
    },
    "bolaji oki|manoel sousa": {
        "fighter_1": "Bolaji Oki", "fighter_2": "Manoel Sousa",
        "odds": {
            "draftkings": {"f1_moneyline": 220, "f2_moneyline": -270},
            "fanduel": {"f1_moneyline": 220, "f2_moneyline": -270},
            "consensus": {"f1_fair": 0.2999, "f2_fair": 0.7001}
        }
    },
    "vitor petrino|steven asplund": {
        "fighter_1": "Vitor Petrino", "fighter_2": "Steven Asplund",
        "odds": {
            "draftkings": {"f1_moneyline": -240, "f2_moneyline": 190},
            "fanduel": {"f1_moneyline": -220, "f2_moneyline": 180},
            "consensus": {"f1_fair": 0.6718, "f2_fair": 0.3282}
        }
    },
    "andre fili|jose delgado": {
        "fighter_1": "Andre Fili", "fighter_2": "Jose Delgado",
        "odds": {
            "draftkings": {"f1_moneyline": 270, "f2_moneyline": -355},
            "fanduel": {"f1_moneyline": 340, "f2_moneyline": -440},
            "consensus": {"f1_fair": 0.2573, "f2_fair": 0.7427}
        }
    },
}


def name_to_slug(name: str) -> str:
    slug = name.lower()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'\s+', '-', slug)
    return slug


def get_initials(name: str) -> str:
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}{parts[-1][0]}".upper()
    return name[:2].upper()


def get_headshot_base64(name: str) -> Tuple[Optional[str], str]:
    slug = name_to_slug(name)
    headshot_path = FIGHTERS_DIR / f"{slug}.png"
    if headshot_path.exists():
        with open(headshot_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{b64}", ""
    return None, get_initials(name)


def load_event_fights(event_id: str) -> Optional[dict]:
    with open(LEDGER_PATH) as f:
        ledger = json.load(f)
    for event in ledger.get('entries', []):
        if event.get('event_id') == event_id:
            return event
    return None


def normalize_name(name: str) -> str:
    import unicodedata
    name = name.lower().strip()
    name = name.replace('.', '').replace("'", '').replace('-', ' ')
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    return ' '.join(name.split())


def get_odds_for_fight(f1: str, f2: str) -> Optional[dict]:
    key = f"{f1.lower()}|{f2.lower()}"
    if key in ODDS_DATA:
        return ODDS_DATA[key]
    key_rev = f"{f2.lower()}|{f1.lower()}"
    if key_rev in ODDS_DATA:
        return ODDS_DATA[key_rev]

    f1_norm = normalize_name(f1)
    f2_norm = normalize_name(f2)
    f1_last = f1_norm.split()[-1] if f1_norm.split() else f1_norm
    f2_last = f2_norm.split()[-1] if f2_norm.split() else f2_norm

    for odds_key, fight in ODDS_DATA.items():
        parts = odds_key.split('|')
        if len(parts) != 2:
            continue
        o1_last = normalize_name(parts[0]).split()[-1]
        o2_last = normalize_name(parts[1]).split()[-1]
        if (f1_last == o1_last and f2_last == o2_last) or \
           (f1_last == o2_last and f2_last == o1_last):
            return fight
    return None


def format_ml(ml: int) -> str:
    if ml > 0:
        return f"+{ml}"
    return str(ml)


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
    font-size: 24px;
    -webkit-font-smoothing: antialiased;
    padding: 36px 0;
    width: 1300px;
    overflow: visible;
}

.fight-list {
    max-width: 1100px;
    margin: 0 auto;
    overflow: visible;
}

.fight-card {
    background: #ffffff;
    padding: 28px 0;
    overflow: visible;
    display: grid;
    grid-template-rows: 28px 110px 84px;
    gap: 0;
}

.fight-card + .fight-card {
    border-top: 1px solid var(--border);
}

.fight-card.main-event {
    padding: 40px 0;
    grid-template-rows: 28px 140px 84px;
}

.fight-card.main-event .fighter-name {
    font-size: 1.6rem;
}

.fight-card.main-event .probability {
    font-size: 2.1rem;
}

.fight-card.main-event .fighter-headshot,
.fight-card.main-event .fighter-headshot-fallback {
    width: 100px;
    height: 100px;
}

.card-section-divider {
    padding: 22px 0;
    text-align: center;
}

.card-section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    background: #ffffff;
    padding: 6px 22px;
    border: 1px dashed var(--border);
}

.fight-position-row {
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: flex-start;
}

.fight-position {
    font-size: 1rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
}

.fight-row {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 24px;
    align-items: center;
    overflow: visible;
    height: 110px;
}

.fight-card.main-event .fight-row {
    height: 140px;
}

.fighter-side {
    display: flex;
    align-items: center;
    gap: 18px;
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

.fighter-info {
    flex-shrink: 0;
    width: 180px;
}

.fighter-headshot {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid var(--border);
    flex-shrink: 0;
}

.fighter-headshot-fallback {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    border: 2px solid var(--border);
    background: var(--code-bg);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    color: var(--text-muted);
    flex-shrink: 0;
}

.fighter-name {
    font-family: 'Fraunces', serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.2;
}

.fighter-meta {
    font-size: 1.05rem;
    color: var(--text-muted);
    margin-top: 3px;
}

.fighter-elo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1rem;
    color: var(--text-muted);
    margin-top: 3px;
}

.fighter-elo.higher {
    color: var(--red);
    font-weight: 500;
}

.fighter-side.left .fighter-elo {
    text-align: right;
}

.prob-display {
    display: flex;
    align-items: center;
    gap: 14px;
    flex-shrink: 0;
}

.fighter-side.left .prob-display {
    flex-direction: row-reverse;
}

.probability {
    font-family: 'Fraunces', serif;
    font-size: 1.75rem;
    font-weight: 700;
    min-width: 80px;
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
    width: 100px;
    height: 10px;
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
    font-size: 1.05rem;
    color: var(--text-muted);
    text-transform: lowercase;
    letter-spacing: 0.05em;
    padding: 0 6px;
}

.odds-row {
    height: 84px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-family: 'IBM Plex Mono', monospace;
}

.odds-moneylines {
    font-size: 1.1rem;
    color: var(--text-muted);
    margin-bottom: 3px;
}

.odds-consensus {
    font-size: 1.2rem;
    color: var(--text-secondary);
    margin-bottom: 3px;
}

.odds-edge {
    font-size: 1.2rem;
    color: var(--text-muted);
}

.odds-edge.value-edge {
    color: var(--red);
    font-weight: 500;
}

.odds-unavailable {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.2rem;
    color: var(--text-muted);
}
"""


def generate_headshot_html(name: str) -> str:
    b64_data, initials = get_headshot_base64(name)
    if b64_data:
        return f'<img src="{b64_data}" alt="{name}" class="fighter-headshot">'
    return f'<div class="fighter-headshot-fallback">{initials}</div>'


def generate_fight_card_div(fight: dict, position_label: str, is_main_event: bool = False) -> str:
    f1 = fight['fighter_1']
    f2 = fight['fighter_2']
    f1_prob = fight['f1_win_prob']
    f2_prob = fight['f2_win_prob']
    f1_pct = f"{f1_prob * 100:.1f}"
    f2_pct = f"{f2_prob * 100:.1f}"
    f1_elo = fight.get('f1_elo', 1500)
    f2_elo = fight.get('f2_elo', 1500)
    weight_class = fight.get('weight_class', '')

    f1_headshot = generate_headshot_html(f1)
    f2_headshot = generate_headshot_html(f2)

    # Elo: red for higher (favored)
    f1_elo_class = 'higher' if f1_elo > f2_elo else ''
    f2_elo_class = 'higher' if f2_elo > f1_elo else ''

    card_class = 'fight-card main-event' if is_main_event else 'fight-card'

    show_label = position_label in ['Main Event', 'Co-Main']
    position_html = position_label if show_label else ''

    # Odds
    odds = get_odds_for_fight(f1, f2)
    if odds:
        odds_data = odds.get('odds', {})
        dk = odds_data.get('draftkings', {})
        fd = odds_data.get('fanduel', {})
        consensus = odds_data.get('consensus', {})

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

        edge = None
        if consensus_f1 is not None:
            edge = f1_prob - consensus_f1

        odds_content = ""
        if moneylines:
            odds_content += f'<div class="odds-moneylines">{" &middot; ".join(moneylines)}</div>'
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
        odds_html = '<div class="odds-row"><span class="odds-unavailable">Odds unavailable</span></div>'

    return f'''<div class="{card_class}">
    <div class="fight-position-row"><span class="fight-position">{position_html}</span></div>

    <div class="fight-row">
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

        <div class="vs-divider">vs</div>

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


async def capture_html_to_png(html_path: Path, output_path: Path, viewport_width: int = 1300):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(
            viewport={"width": viewport_width, "height": 800},
            device_scale_factor=2
        )
        await page.goto(f"file://{html_path.absolute()}")
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(500)
        await page.screenshot(path=str(output_path), full_page=True)
        await browser.close()


async def main():
    print("Generating combined fight cards for Substack...")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    event = load_event_fights(EVENT_ID)
    if not event:
        print(f"ERROR: Event '{EVENT_ID}' not found in ledger")
        return

    fights = event.get('fights', [])

    print(f"Found {len(fights)} fights")
    print(f"Event: {event.get('event_name', 'Unknown')}")
    print(f"Date: {event.get('event_date', 'Unknown')}")

    # Split: main card = first 4 fights, prelims = rest
    main_card_fights = fights[:MAIN_CARD_COUNT]
    prelim_fights = fights[MAIN_CARD_COUNT:]

    print(f"Main card: {len(main_card_fights)} fights")
    print(f"Prelims: {len(prelim_fights)} fights")

    # Check odds coverage
    for fight in fights:
        f1, f2 = fight['fighter_1'], fight['fighter_2']
        has_odds = get_odds_for_fight(f1, f2) is not None
        status = "OK" if has_odds else "NO ODDS"
        print(f"  [{status}] {f1} vs {f2}")

    # 1. Hero image (main event only)
    print("\n1. Generating hero image...")
    hero_content = generate_fight_card_div(fights[0], "Main Event", is_main_event=True)
    hero_html = wrap_in_html(hero_content, "Main Event: Emmett vs Vallejos")
    hero_html_path = OUTPUT_DIR / "hero_emmett_vallejos.html"
    with open(hero_html_path, 'w') as f:
        f.write(hero_html)
    print(f"  Created: {hero_html_path}")

    # 2. Main card
    print("\n2. Generating main card image...")
    main_card_content = ""
    for i, fight in enumerate(main_card_fights):
        is_main = (i == 0)
        if is_main:
            label = "Main Event"
        elif i == 1:
            label = "Co-Main"
        else:
            label = ""
        main_card_content += generate_fight_card_div(fight, label, is_main)
        main_card_content += "\n"

    main_card_html = wrap_in_html(main_card_content, "Main Card")
    main_card_html_path = OUTPUT_DIR / "main_card.html"
    with open(main_card_html_path, 'w') as f:
        f.write(main_card_html)
    print(f"  Created: {main_card_html_path}")

    # 3. Prelim card
    print("\n3. Generating prelim card image...")
    prelim_content = '<div class="card-section-divider"><span class="card-section-label">Preliminary Card</span></div>\n'
    for fight in prelim_fights:
        prelim_content += generate_fight_card_div(fight, "", False)
        prelim_content += "\n"

    prelim_html = wrap_in_html(prelim_content, "Preliminary Card")
    prelim_html_path = OUTPUT_DIR / "prelim_card.html"
    with open(prelim_html_path, 'w') as f:
        f.write(prelim_html)
    print(f"  Created: {prelim_html_path}")

    # Convert to PNG
    print("\n" + "-" * 60)
    print("Converting to PNG...")

    await capture_html_to_png(hero_html_path, OUTPUT_DIR / "hero_emmett_vallejos.png")
    print("  Captured: hero_emmett_vallejos.png")

    await capture_html_to_png(main_card_html_path, OUTPUT_DIR / "main_card.png")
    print("  Captured: main_card.png")

    await capture_html_to_png(prelim_html_path, OUTPUT_DIR / "prelim_card.png")
    print("  Captured: prelim_card.png")

    print("\n" + "=" * 60)
    print("Done!")

    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        size = f.stat().st_size
        print(f"  {f.name}: {size:,} bytes")


if __name__ == "__main__":
    asyncio.run(main())
