import logging

import torch
from pysentimiento import create_analyzer

from ..domain.enums.sentiment_label import SentimentLabel
from ..domain.repositories.sentiment_analyzer import SentimentAnalyzer
from ..domain.value_objects.sentiment_result import SentimentResult

logger = logging.getLogger(__name__)

_LABEL_MAP = {
    "POS": SentimentLabel.POSITIVE,
    "NEU": SentimentLabel.NEUTRAL,
    "NEG": SentimentLabel.NEGATIVE,
}

_FALLBACK = SentimentResult(score=0.0, label=SentimentLabel.NEUTRAL)


class PySentimientoAnalyzer(SentimentAnalyzer):
    """
    Adapter que usa o modelo pysentimiento/bertweet-pt-sentiment para
    realizar análise de sentimento em português de forma totalmente local,
    sem dependência de APIs externas ou limites de rate.

    Usa GPU (CUDA) automaticamente se disponível, caso contrário usa CPU.
    O modelo é carregado uma única vez no startup e mantido em memória.
    Score derivado: probas[POS] - probas[NEG], no intervalo [-1.0, 1.0].
    """

    def __init__(self):
        if torch.cuda.is_available():
            self._device = 0  # índice da primeira GPU (RTX 4050)
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detectada: {device_name}. Usando CUDA para inferência.")
        else:
            self._device = -1  # CPU
            logger.warning("GPU não disponível. Usando CPU para inferência.")

        logger.info("Carregando modelo pysentimiento/bertweet-pt-sentiment...")
        self._model = create_analyzer(task="sentiment", lang="pt", model_kwargs={"device": self._device})
        logger.info("Modelo carregado com sucesso.")

    def analyze(self, sentiment: str) -> SentimentResult:
        try:
            result = self._model.predict(sentiment)

            label = _LABEL_MAP.get(result.output, SentimentLabel.NEUTRAL)

            probas = result.probas
            score = round(probas.get("POS", 0.0) - probas.get("NEG", 0.0), 4)

            return SentimentResult(score=score, label=label)

        except Exception:
            logger.exception(
                "Erro ao analisar sentimento com pysentimiento. Retornando NEUTRAL."
            )
            return _FALLBACK
