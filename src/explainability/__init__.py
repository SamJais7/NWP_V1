"""
Explainable AI (XAI) and Forecaster Decision Support Modules.
"""

from src.explainability.explainer import ContextAwareExplainer, SynopticPatternClassifier
from src.explainability.analog_finder import PrecedentAnalogFinder
from src.explainability.templates import AdvisoryGenerator

__all__ = [
    "ContextAwareExplainer",
    "SynopticPatternClassifier",
    "PrecedentAnalogFinder",
    "AdvisoryGenerator"
]
