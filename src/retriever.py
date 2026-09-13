"""
Retrieval-for-grounding: given a new customer message, find the most similar
historically-RESOLVED threads so the reply generator can ground its draft in
what actually worked before, instead of hallucinating a policy.

Deliberately simple: TF-IDF + cosine similarity over resolved customer_text,
scoped to the target brand's own history. No vector DB, no embeddings API —
this is a decision documented in DECISION_LOG.md #5 (not because embeddings
wouldn't be better, but because TF-IDF is free, deterministic, and good
enough at this corpus size; the eval harness would tell us if it weren't).
"""

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import Thread
from src import config


@dataclass
class RetrievedResolution:
    customer_text: str
    agent_reply: str
    similarity: float


class ResolutionRetriever:
    def __init__(self, threads: list[Thread]):
        self.resolved = [t for t in threads if t.was_resolved and not t.had_repeat_contact]
        if not self.resolved:
            raise ValueError("No resolved, non-repeat-contact threads to build retriever from.")
        self.vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        self.matrix = self.vectorizer.fit_transform([t.customer_text for t in self.resolved])

    def retrieve(self, query_text: str, top_k: int = None) -> list[RetrievedResolution]:
        top_k = top_k or config.RETRIEVAL_TOP_K
        q_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        ranked_idx = sims.argsort()[::-1][:top_k]
        return [
            RetrievedResolution(
                customer_text=self.resolved[i].customer_text,
                agent_reply=self.resolved[i].agent_reply,
                similarity=float(sims[i]),
            )
            for i in ranked_idx
            if sims[i] > 0
        ]
