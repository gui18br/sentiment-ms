import re
import math
from typing import List, Dict
from dataclasses import dataclass

from ..domain.entities.sentiment import Sentiment
from ..domain.enums.sentiment_label import SentimentLabel
from ..domain.repositories.sentiment_analyzer import SentimentAnalyzer
from ..domain.value_objects.sentiment_result import SentimentResult


@dataclass
class LexiconEntry:
    weight: float
    terms: List[str]


class SimpleSentimentAnalyzer(SentimentAnalyzer):

    # ==============================
    # Configurações
    # ==============================

    NEGATION_WINDOW = 3
    POSITION_WEIGHT_FACTOR = 0.15
    INTENSIFIER_MULTIPLIER = 1.8
    DIMINISHER_MULTIPLIER = 0.6
    EMBEDDING_DIMENSION = 256

    # ==============================
    # Stopwords
    # ==============================

    stopwords = {
        "o","a","os","as","um","uma","de","da","do","e","é",
        "em","para","com","que","isso","isto","no","na"
    }

    # ==============================
    # Negação & Intensidade
    # ==============================

    negations = ["não", "nunca", "jamais", "nem"]
    intensifiers = ["muito", "extremamente", "super", "absurdamente"]
    diminishers = ["pouco", "levemente", "meio"]

    # ==============================
    # Failure verbs
    # ==============================

    failure_verbs = ["quebrou", "travou", "bugou", "caiu", "parou"]

    # ==============================
    # Lexicons
    # ==============================

    negative_lexicon = [
        LexiconEntry(-3, ["péssimo", "horrível", "inaceitável"]),
        LexiconEntry(-2, ["ruim", "bugado", "erro", "travando"]),
        LexiconEntry(-1, ["lento", "confuso", "estranho"]),
    ]

    positive_lexicon = [
        LexiconEntry(3, ["excelente", "perfeito", "impecável"]),
        LexiconEntry(2, ["bom", "funciona", "ótimo"]),
        LexiconEntry(1, ["aceitável", "ok", "normal"]),
    ]

    # ==============================
    # PIPELINE
    # ==============================

    def analyze(self, sentiment: Sentiment) -> SentimentResult:
        normalized = self.normalize_text(sentiment.text)

        tokens = self.tokenize(normalized)
        filtered = self.remove_stopwords(tokens)
        stemmed = self.stem_tokens(filtered)

        ngrams = self.generate_ngrams(stemmed)

        score = 0
        score += self.lexicon_score(stemmed)
        score += self.failure_verb_score(stemmed)
        score += self.negation_score(stemmed)
        score += self.intensity_score(stemmed)
        score += self.frequency_score(stemmed)
        score += self.ngram_score(ngrams)
        score += self.embedding_similarity_score(stemmed)

        normalized_score = score / max(len(stemmed), 1)

        label = SentimentLabel.NEUTRAL
        if normalized_score > 0.05:
            label = SentimentLabel.POSITIVE
        if normalized_score < -0.05:
            label = SentimentLabel.NEGATIVE

        return SentimentResult(
            score=round(normalized_score, 3),
            label=label,
        )

    # ==============================
    # 1. Normalização
    # ==============================

    def normalize_text(self, text: str) -> str:
        text = text.lower()

        text = (
            text.encode("ascii", "ignore")
            .decode("utf-8")
        )

        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ==============================
    # 2. Tokenização
    # ==============================

    def tokenize(self, text: str) -> List[str]:
        return text.split(" ")

    # ==============================
    # 3. Stopwords
    # ==============================

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in self.stopwords]

    # ==============================
    # 4. Stemming simples
    # ==============================

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        stemmed = []

        for token in tokens:
            token = re.sub(r"mente$", "", token)
            token = re.sub(r"coes$", "cao", token)
            token = re.sub(r"s$", "", token)
            stemmed.append(token)

        return stemmed

    # ==============================
    # 5. N-grams
    # ==============================

    def generate_ngrams(self, tokens: List[str]) -> List[str]:
        ngrams = []

        for i in range(len(tokens)):
            if i + 1 < len(tokens):
                ngrams.append(f"{tokens[i]} {tokens[i+1]}")
            if i + 2 < len(tokens):
                ngrams.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")

        return ngrams

    # ==============================
    # 6. Lexicon score
    # ==============================

    def lexicon_score(self, tokens: List[str]) -> float:
        score = 0

        for index, token in enumerate(tokens):
            position_weight = 1 + (
                (len(tokens) - index) * self.POSITION_WEIGHT_FACTOR
            )

            for entry in self.positive_lexicon:
                if token in entry.terms:
                    score += entry.weight * position_weight

            for entry in self.negative_lexicon:
                if token in entry.terms:
                    score += entry.weight * position_weight

        return score

    # ==============================
    # 7. Failure verbs
    # ==============================

    def failure_verb_score(self, tokens: List[str]) -> float:
        score = 0

        for t in tokens:
            if t in self.failure_verbs:
                score -= 2.5

        return score

    # ==============================
    # 8. Negação
    # ==============================

    def negation_score(self, tokens: List[str]) -> float:
        score = 0

        for index, token in enumerate(tokens):
            if token in self.negations:
                for i in range(1, self.NEGATION_WINDOW + 1):
                    if index + i >= len(tokens):
                        continue

                    next_token = tokens[index + i]

                    if self.is_positive(next_token):
                        score -= 2
                    if self.is_negative(next_token):
                        score += 2

        return score

    # ==============================
    # 9. Intensidade
    # ==============================

    def intensity_score(self, tokens: List[str]) -> float:
        score = 0

        for index, token in enumerate(tokens):
            if index + 1 >= len(tokens):
                continue

            next_token = tokens[index + 1]

            if token in self.intensifiers:
                if self.is_positive(next_token):
                    score += self.INTENSIFIER_MULTIPLIER
                if self.is_negative(next_token):
                    score -= self.INTENSIFIER_MULTIPLIER

            if token in self.diminishers:
                if self.is_positive(next_token):
                    score += self.DIMINISHER_MULTIPLIER
                if self.is_negative(next_token):
                    score -= self.DIMINISHER_MULTIPLIER

        return score

    # ==============================
    # 10. Frequência
    # ==============================

    def frequency_score(self, tokens: List[str]) -> float:
        freq: Dict[str, int] = {}

        for t in tokens:
            freq[t] = freq.get(t, 0) + 1

        score = 0

        for term, count in freq.items():
            if self.is_positive(term):
                score += count * 0.5
            if self.is_negative(term):
                score -= count * 0.5

        return score

    # ==============================
    # 11. N-gram score
    # ==============================

    def ngram_score(self, ngrams: List[str]) -> float:
        score = 0

        for g in ngrams:
            if "nao bom" in g:
                score -= 2
            if "muito bom" in g:
                score += 2
            if "nao funciona" in g:
                score -= 3

        return score

    # ==============================
    # 12. Embedding fake
    # ==============================

    def embedding_similarity_score(self, tokens: List[str]) -> float:
        text_vector = self.fake_embedding(" ".join(tokens))

        positive_anchor = self.fake_embedding(
            "excelente perfeito otimo"
        )
        negative_anchor = self.fake_embedding(
            "ruim pessimo horrivel"
        )

        pos_sim = self.cosine_similarity(text_vector, positive_anchor)
        neg_sim = self.cosine_similarity(text_vector, negative_anchor)

        return (pos_sim - neg_sim) * 5

    def fake_embedding(self, text: str) -> List[float]:
        vector = [0] * self.EMBEDDING_DIMENSION

        for i, char in enumerate(text):
            vector[i % self.EMBEDDING_DIMENSION] += ord(char)

        return vector

    def cosine_similarity(
        self, a: List[float], b: List[float]
    ) -> float:
        dot = 0
        norm_a = 0
        norm_b = 0

        for i in range(len(a)):
            dot += a[i] * b[i]
            norm_a += a[i] ** 2
            norm_b += b[i] ** 2

        denom = math.sqrt(norm_a) * math.sqrt(norm_b)
        return dot / (denom if denom else 1)

    # ==============================
    # Helpers
    # ==============================

    def is_positive(self, term: str) -> bool:
        return any(term in l.terms for l in self.positive_lexicon)

    def is_negative(self, term: str) -> bool:
        return any(term in l.terms for l in self.negative_lexicon)