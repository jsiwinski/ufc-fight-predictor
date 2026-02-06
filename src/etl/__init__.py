"""
ETL (Extract, Transform, Load) module for UFC fight data.

This module handles:
- Data collection from various sources
- Data cleaning and validation
- Data transformation and storage
- Regular updates for upcoming events
"""

from .scraper import UFCDataScraper

__all__ = ['UFCDataScraper']
