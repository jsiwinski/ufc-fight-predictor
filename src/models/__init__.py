"""
Machine learning models module for UFC fight predictions.

This module handles:
- Model training and validation
- Model evaluation and metrics
- Prediction generation
- Model persistence
"""

from .train import ModelTrainer
from .predict import FightPredictor

__all__ = ['ModelTrainer', 'FightPredictor']
