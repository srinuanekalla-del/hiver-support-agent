"""
Thin re-export module. The actual implementations live in src/classifier.py
next to LLMClassifier, since all three share the same ClassificationResult
interface and it made more sense to keep classifiers together (see
DECISION_LOG.md #4). This file exists so the two required baselines are easy
to find from the project root without hunting through src/.
"""

from src.classifier import TrivialBaseline, KeywordBaseline  # noqa: F401

__all__ = ["TrivialBaseline", "KeywordBaseline"]
