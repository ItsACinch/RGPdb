"""Caller-side reference schedule predictor: question text -> relation schedule.

Reference/test implementation only (the engine is predictor-agnostic; production
callers supply their own schedules). Uses an entity-masked TF-IDF + logistic
classifier to map a question to its MetaQA qtype, then qtype_to_relation_sequence to a
relation-id schedule.

CAVEAT: MetaQA questions are templated, so this classifier is near-perfect here; that
is NOT evidence that real open-ended question->schedule prediction is easy.
"""
from __future__ import annotations
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .metaqa import qtype_to_relation_sequence

_ENT = re.compile(r"\[.*?\]")


def _mask(question: str) -> str:
    return _ENT.sub(" ENT ", question)


class SchedulePredictor:
    """Fit on (question, qtype) pairs; predict a relation-id schedule for a question."""

    def __init__(self, relation_to_id: dict[str, int]):
        self.relation_to_id = relation_to_id
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
            ("clf", LogisticRegression(max_iter=1000)),
        ])

    def fit(self, questions: list[str], qtypes: list[str]) -> "SchedulePredictor":
        self.model.fit([_mask(q) for q in questions], qtypes)
        return self

    def predict_qtype(self, question: str) -> str:
        return self.model.predict([_mask(question)])[0]

    def predict_schedule(self, question: str) -> list[int]:
        seq = qtype_to_relation_sequence(self.predict_qtype(question))
        return [self.relation_to_id[r] for r in seq if r in self.relation_to_id]
