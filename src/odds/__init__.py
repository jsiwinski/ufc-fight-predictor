"""
UFC Moneyline Odds Scraper Module.

Scrapes moneyline odds from sportsbooks (via The Odds API) for UFC fights.
Provides probability conversion, fighter name matching, and value bet identification.
"""

from .scraper import (
    UFCOddsScraper,
    american_to_implied_prob,
    remove_vig,
)

__all__ = [
    'UFCOddsScraper',
    'american_to_implied_prob',
    'remove_vig',
]
