"""
Feature engineering module for UFC fight predictions.

This module handles:
- Fighter statistics calculation
- Matchup feature generation
- Historical performance features
- Feature selection and transformation
"""

from .engineer import FeatureEngineer

__all__ = ['FeatureEngineer']
