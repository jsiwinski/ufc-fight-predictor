"""
Web application module for displaying UFC fight predictions.

This module handles:
- Web interface for displaying predictions
- Real-time prediction updates
- Fight card visualization
- Prediction history tracking
"""

from .app import create_app

__all__ = ['create_app']
