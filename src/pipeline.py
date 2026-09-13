"""
End-to-end pipeline: classify -> retrieve grounding -> draft reply -> decide
auto-handle vs escalate.

This is the single object the eval harness, the CLI demo, and (in
principle) a real-time webhook handler would all call.
"""

from dataclasses import dataclass

from src.classifier import LLMClassifier, ClassificationResult
from src.retriever import ResolutionRetriever, RetrievedResolution
from src.reply_generator import ReplyGenerator, DraftReply
from src.decision_engine import DecisionEngine, Decision
from src.data_loader import load_threads


@dataclass
class PipelineResult:
    message: str
    classification: ClassificationResult
    retrieved: list[RetrievedResolution]
    draft_reply: DraftReply
    decision: Decision


class SupportAgentPipeline:
    def __init__(self, threads=None):
        self.threads = threads if threads is not None else load_threads()
        self.classifier = LLMClassifier()
        self.retriever = ResolutionRetriever(self.threads)
        self.reply_generator = ReplyGenerator()
        self.decision_engine = DecisionEngine()

    def process(self, message: str, repeat_contact: bool = False) -> PipelineResult:
        classification = self.classifier.classify(message)
        retrieved = self.retriever.retrieve(message)
        draft = self.reply_generator.generate(message, retrieved)
        decision = self.decision_engine.decide(
            message, classification.intent, classification.confidence, repeat_contact
        )
        return PipelineResult(
            message=message,
            classification=classification,
            retrieved=retrieved,
            draft_reply=draft,
            decision=decision,
        )
