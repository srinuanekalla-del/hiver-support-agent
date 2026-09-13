import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import SupportAgentPipeline
from src.decision_engine import DecisionEngine
from src.classifier import KeywordBaseline, TrivialBaseline


def test_pipeline_runs_end_to_end():
    pipeline = SupportAgentPipeline()
    result = pipeline.process("@PixelWaveHelp my hub won't turn on")
    assert result.classification.intent
    assert result.decision.action in ("auto_handle", "escalate")
    assert result.draft_reply.text


def test_hard_escalation_phrase_always_escalates():
    engine = DecisionEngine()
    d = engine.decide("thinking about a lawsuit here @PixelWaveHelp", "device_technical_issue", 0.99)
    assert d.action == "escalate"


def test_low_confidence_forces_escalation():
    engine = DecisionEngine()
    d = engine.decide("some ambiguous message", "device_technical_issue", 0.1)
    assert d.action == "escalate"


def test_refund_intent_always_escalates_even_high_confidence():
    engine = DecisionEngine()
    d = engine.decide("please refund me @PixelWaveHelp", "cancellation_refund_request", 0.99)
    assert d.action == "escalate"


def test_keyword_baseline_returns_valid_intent():
    from src.intents import INTENT_KEYS

    result = KeywordBaseline().classify("i want a refund please")
    assert result.intent in INTENT_KEYS


def test_trivial_baseline_always_same():
    clf = TrivialBaseline("device_technical_issue")
    assert clf.classify("anything at all").intent == "device_technical_issue"
    assert clf.classify("something else").intent == "device_technical_issue"


if __name__ == "__main__":
    test_pipeline_runs_end_to_end()
    test_hard_escalation_phrase_always_escalates()
    test_low_confidence_forces_escalation()
    test_refund_intent_always_escalates_even_high_confidence()
    test_keyword_baseline_returns_valid_intent()
    test_trivial_baseline_always_same()
    print("All tests passed.")
