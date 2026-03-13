import re
import math
from typing import List, Dict
from dataclasses import dataclass

from ..domain.enums.sentiment_label import SentimentLabel
from ..domain.repositories.sentiment_analyzer import SentimentAnalyzer
from ..domain.value_objects.sentiment_result import SentimentResult


@dataclass
class LexiconEntry:
    weight: float
    terms: List[str]


class AdvancedSentimentAnalyzer(SentimentAnalyzer):

    # ==============================
    # Configurações
    # ==============================

    NEGATION_WINDOW = 4
    POSITION_WEIGHT_FACTOR = 0.15
    VECTOR_DIM = 1536

    # ==============================
    # STOPWORDS
    # ==============================

    stopwords = {
        "o","a","os","as","um","uma","de","da","do","e","é",
        "em","para","com","que","isso","isto","no","na",
        "nos","nas","por","como","mais","menos"
    }

    # ==============================
    # MODIFIERS
    # ==============================

    negations = ["nao","nunca","jamais","nem"]

    intensifiers = [
        "muito","extremamente","super",
        "absurdamente","bastante"
    ]

    diminishers = [
        "pouco","levemente","meio","relativamente"
    ]

    failure_verbs = [
        "quebrou","travou","bugou","caiu",
        "parou","falhou","congelou"
    ]

    # ==============================
    # LEXICONS
    # ==============================

    negative_lexicon = [

        LexiconEntry(-3, [
            "pessimo","horrivel","inaceitavel","frustrante"
        ]),

        LexiconEntry(-2, [
            "ruim","erro","falha","falhas","problema",
            "problemas","bug","bugs","travamento",
            "travamentos","instabilidade","degradacao",
            "lentidao","perda","inconsistencia"
        ]),

        LexiconEntry(-1, [
            "lento","confuso","demorado",
            "complexo","limitante"
        ])
    ]

    positive_lexicon = [

        LexiconEntry(3, [
            "excelente","perfeito","impecavel"
        ]),

        LexiconEntry(2, [
            "bom","funciona","otimo","rapido"
        ]),

        LexiconEntry(1, [
            "aceitavel","ok","normal","estavel"
        ])
    ]

    # ==============================
    # COMPLAINT EXPRESSIONS
    # ==============================

    complaint_expressions = [
        "oportunidade de melhoria",
        "ainda existem",
        "continua apresentando",
        "ainda apresenta",
        "nao esta otimizado",
        "tempo de resposta alto",
        "mensagem de erro",
        "perda de informacao",
        "ficar sem resposta"
    ]

    # ==============================
    # PIPELINE
    # ==============================

    def analyze(self, sentiment: str) -> SentimentResult:

        normalized = self.normalize_text(sentiment)

        tokens = self.tokenize(normalized)
        filtered = self.remove_stopwords(tokens)
        stemmed = self.stem_tokens(filtered)

        ngrams = self.generate_ngrams(stemmed)

        score = 0

        score += self.lexicon_score(stemmed)
        score += self.negation_score(stemmed)
        score += self.intensity_score(stemmed)
        score += self.failure_verb_score(stemmed)
        score += self.tfidf_score(stemmed)
        score += self.ngram_score(ngrams)
        score += self.syntactic_score(stemmed)
        score += self.semantic_distance_score(stemmed)
        score += self.embedding_score(stemmed)

        graph = self.build_cooccurrence_graph(stemmed)

        score += self.sentiment_propagation(graph)

        ranks = self.page_rank(graph)

        score += self.graph_rank_sentiment(ranks)

        score += self.global_semantic_similarity(stemmed)

        score += self.complaint_score(normalized)

        normalized_score = score / math.sqrt(len(stemmed) + 1)

        label = SentimentLabel.NEUTRAL

        if normalized_score > 0.2:
            label = SentimentLabel.POSITIVE

        if normalized_score < -0.2:
            label = SentimentLabel.NEGATIVE

        return SentimentResult(
            score=round(normalized_score,3),
            label=label
        )

    # ==============================
    # NORMALIZATION
    # ==============================

    def normalize_text(self, text: str) -> str:

        text = text.lower()

        text = (
            text.encode("ascii","ignore")
            .decode("utf-8")
        )

        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        return text.split(" ")

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in self.stopwords]

    def stem_tokens(self, tokens: List[str]) -> List[str]:

        stemmed = []

        for token in tokens:
            token = re.sub(r"mente$", "", token)
            token = re.sub(r"coes$", "cao", token)
            token = re.sub(r"s$", "", token)
            stemmed.append(token)

        return stemmed

    # ==============================
    # NGRAMS
    # ==============================

    def generate_ngrams(self, tokens: List[str]) -> List[str]:

        ngrams = []

        for i in range(len(tokens)):

            if i + 1 < len(tokens):
                ngrams.append(f"{tokens[i]} {tokens[i+1]}")

            if i + 2 < len(tokens):
                ngrams.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")

        return ngrams

    def ngram_score(self, ngrams: List[str]) -> float:

        score = 0

        for g in ngrams:

            if "nao bom" in g:
                score -= 2

            if "muito bom" in g:
                score += 2

            if "nao funciona" in g:
                score -= 3

            if "tempo resposta alto" in g:
                score -= 2

            if "mensagem erro" in g:
                score -= 2

            if "perda informacao" in g:
                score -= 3

        return score

    # ==============================
    # POS TAG
    # ==============================

    def pos_tag(self, token: str) -> str:

        if token.endswith("mente"):
            return "ADV"

        if token.endswith("ar") or token.endswith("er") or token.endswith("ir"):
            return "VERB"

        if self.is_positive(token) or self.is_negative(token):
            return "ADJ"

        return "NOUN"

    def syntactic_score(self, tokens: List[str]) -> float:

        score = 0

        for token in tokens:

            tag = self.pos_tag(token)

            if tag == "ADJ":

                if self.is_positive(token):
                    score += 1.2

                if self.is_negative(token):
                    score -= 1.2

        return score

    # ==============================
    # LEXICON
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
    # FAILURE VERBS
    # ==============================

    def failure_verb_score(self, tokens: List[str]) -> float:

        score = 0

        for t in tokens:
            if t in self.failure_verbs:
                score -= 2.5

        return score

    # ==============================
    # NEGATION
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
    # INTENSITY
    # ==============================

    def intensity_score(self, tokens: List[str]) -> float:

        score = 0

        for index, token in enumerate(tokens):

            if index + 1 >= len(tokens):
                continue

            next_token = tokens[index + 1]

            if token in self.intensifiers:

                if self.is_positive(next_token):
                    score += 1.8

                if self.is_negative(next_token):
                    score -= 1.8

            if token in self.diminishers:

                if self.is_positive(next_token):
                    score += 0.6

                if self.is_negative(next_token):
                    score -= 0.6

        return score

    # ==============================
    # TFIDF
    # ==============================

    def tfidf_score(self, tokens: List[str]) -> float:

        freq: Dict[str,int] = {}

        for t in tokens:
            freq[t] = freq.get(t,0) + 1

        score = 0

        for term, count in freq.items():

            tf = count / len(tokens)
            idf = math.log(10000 / (1 + count))

            weight = tf * idf

            if self.is_positive(term):
                score += weight

            if self.is_negative(term):
                score -= weight

        return score

    # ==============================
    # GRAPH
    # ==============================

    def build_cooccurrence_graph(self, tokens: List[str]):

        graph: Dict[str, Dict[str,int]] = {}

        for i in range(len(tokens)):
            for j in range(i+1, len(tokens)):

                a = tokens[i]
                b = tokens[j]

                if a not in graph:
                    graph[a] = {}

                if b not in graph:
                    graph[b] = {}

                graph[a][b] = graph[a].get(b,0) + 1

        return graph

    def page_rank(self, graph, iterations=30):

        ranks = {node:1 for node in graph}

        for _ in range(iterations):

            new_ranks = {}

            for node, edges in graph.items():

                sum_rank = 0

                for neighbor in edges:

                    sum_rank += ranks.get(neighbor,1) / max(len(edges),1)

                new_ranks[node] = 0.15 + 0.85 * sum_rank

            ranks = new_ranks

        return ranks

    def graph_rank_sentiment(self, ranks):

        score = 0

        for token, rank in ranks.items():

            if self.is_positive(token):
                score += rank

            if self.is_negative(token):
                score -= rank

        return score

    def sentiment_propagation(self, graph):

        score = 0

        for node, edges in graph.items():

            for neighbor in edges:

                if self.is_positive(node) and self.is_positive(neighbor):
                    score += 0.5

                if self.is_negative(node) and self.is_negative(neighbor):
                    score -= 0.5

        return score

    # ==============================
    # SEMANTIC
    # ==============================

    def global_semantic_similarity(self, tokens: List[str]) -> float:

        score = 0

        vectors = [self.token_vector(t) for t in tokens]

        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):

                score += self.cosine_similarity(vectors[i], vectors[j])

        return score / (len(tokens) if tokens else 1)

    def token_vector(self, token: str):

        vec = [0] * self.VECTOR_DIM

        for char in token:

            idx = (ord(char) * 31) % self.VECTOR_DIM
            vec[idx] += 1

        return vec

    def sentence_vector(self, tokens: List[str]):

        vec = [0] * self.VECTOR_DIM

        for token in tokens:

            tv = self.token_vector(token)

            for i in range(self.VECTOR_DIM):
                vec[i] += tv[i]

        return vec

    def embedding_score(self, tokens: List[str]):

        text_vec = self.sentence_vector(tokens)

        pos_vec = self.sentence_vector(["excelente","perfeito","otimo"])
        neg_vec = self.sentence_vector(["ruim","pessimo","horrivel"])

        pos_sim = self.cosine_similarity(text_vec,pos_vec)
        neg_sim = self.cosine_similarity(text_vec,neg_vec)

        return (pos_sim - neg_sim) * 2

    def cosine_similarity(self,a,b):

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
    # SEMANTIC DISTANCE
    # ==============================

    def semantic_distance_score(self, tokens: List[str]):

        score = 0

        for i in range(len(tokens)):
            for j in range(i+1,len(tokens)):

                dist = self.levenshtein(tokens[i],tokens[j])

                if dist <= 2:

                    if self.is_positive(tokens[i]):
                        score += 0.2

                    if self.is_negative(tokens[i]):
                        score -= 0.2

        return score

    # ==============================
    # COMPLAINT
    # ==============================

    def complaint_score(self,text:str):

        score = 0

        for expr in self.complaint_expressions:

            if expr in text:
                score -= 2

        return score

    # ==============================
    # LEVENSHTEIN
    # ==============================

    def levenshtein(self,a,b):

        matrix = [[0]*(len(a)+1) for _ in range(len(b)+1)]

        for i in range(len(b)+1):
            matrix[i][0] = i

        for j in range(len(a)+1):
            matrix[0][j] = j

        for i in range(1,len(b)+1):
            for j in range(1,len(a)+1):

                if b[i-1] == a[j-1]:
                    matrix[i][j] = matrix[i-1][j-1]
                else:
                    matrix[i][j] = min(
                        matrix[i-1][j-1]+1,
                        matrix[i][j-1]+1,
                        matrix[i-1][j]+1
                    )

        return matrix[len(b)][len(a)]

    # ==============================
    # HELPERS
    # ==============================

    def is_positive(self,term:str) -> bool:
        return any(term in l.terms for l in self.positive_lexicon)

    def is_negative(self,term:str) -> bool:
        return any(term in l.terms for l in self.negative_lexicon)