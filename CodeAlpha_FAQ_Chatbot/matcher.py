"""
matcher.py
----------
Loads a FAQ dataset and matches incoming user questions against it using
TF-IDF vectorization + cosine similarity (scikit-learn), on top of the
NLTK-based preprocessing in preprocess.py.
"""

import json
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess


@dataclass
class MatchResult:
    question: str
    answer: str
    score: float
    matched: bool


class FAQMatcher:
    def __init__(self, faq_path: str, threshold: float = 0.25):
        """
        faq_path: path to a JSON file: a list of {"question": ..., "answer": ...}
        threshold: minimum cosine similarity required to consider a match
                   confident. Below this, the chatbot admits it doesn't know.
        """
        self.threshold = threshold
        self.faqs = self._load_faqs(faq_path)

        self.questions = [f["question"] for f in self.faqs]
        self.answers = [f["answer"] for f in self.faqs]

        # Preprocess every FAQ question once, up front.
        self._processed_questions = [preprocess(q) for q in self.questions]

        # Fit TF-IDF on the FAQ corpus. ngram_range=(1,2) lets it catch
        # short two-word phrases ("return policy", "track order") too.
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self._faq_matrix = self.vectorizer.fit_transform(self._processed_questions)

    @staticmethod
    def _load_faqs(path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not data:
            raise ValueError("FAQ file must contain a non-empty list of Q&A objects.")
        return data

    def get_best_match(self, user_query: str) -> MatchResult:
        processed_query = preprocess(user_query)

        # Guard against an empty query after preprocessing (e.g. "???").
        if not processed_query.strip():
            return MatchResult(question="", answer="", score=0.0, matched=False)

        query_vec = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vec, self._faq_matrix)[0]

        best_idx = similarities.argmax()
        best_score = float(similarities[best_idx])

        matched = best_score >= self.threshold
        return MatchResult(
            question=self.questions[best_idx],
            answer=self.answers[best_idx],
            score=best_score,
            matched=matched,
        )

    def top_matches(self, user_query: str, k: int = 3):
        """Returns the top-k (question, answer, score) tuples, for debugging
        or for showing 'did you mean' suggestions in the UI."""
        processed_query = preprocess(user_query)
        if not processed_query.strip():
            return []

        query_vec = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vec, self._faq_matrix)[0]
        ranked = similarities.argsort()[::-1][:k]
        return [
            (self.questions[i], self.answers[i], float(similarities[i]))
            for i in ranked
        ]


FALLBACK_RESPONSE = (
    "I'm not confident I have an answer for that yet. "
    "Could you rephrase your question, or contact support@example.com for help?"
)


if __name__ == "__main__":
    matcher = FAQMatcher("faqs.json")
    for q in [
        "When will my stuff arrive?",
        "can i send this plant back",
        "why are my plant's leaves turning yellow",
        "what's the meaning of life",
    ]:
        result = matcher.get_best_match(q)
        print(f"\nUser: {q}")
        if result.matched:
            print(f"Matched FAQ ({result.score:.2f}): {result.question}")
            print(f"Answer: {result.answer}")
        else:
            print(f"No confident match (best score {result.score:.2f})")
            print(f"Answer: {FALLBACK_RESPONSE}")
