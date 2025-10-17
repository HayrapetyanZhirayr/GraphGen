from dataclasses import dataclass, field
from typing import Set

from graphgen.bases.datatypes import QAPair
from graphgen.models.evaluate.base_evaluator import BaseEvaluator
from graphgen.utils import NLTKHelper, create_event_loop, detect_main_language

nltk_helper = NLTKHelper()


@dataclass
class MTLDEvaluator(BaseEvaluator):
    """
    Показатели для измерения разнообразия текстовой лексики
    """

    stopwords_en: Set[str] = field(
        default_factory=lambda: set(nltk_helper.get_stopwords("english"))
    )
    stopwords_zh: Set[str] = field(
        default_factory=lambda: set(nltk_helper.get_stopwords("chinese"))
    )

    async def evaluate_single(self, pair: QAPair) -> float:
        loop = create_event_loop()
        return await loop.run_in_executor(None, self._calculate_mtld_score, pair.answer)

    def _calculate_mtld_score(self, text: str, threshold=0.72) -> float:
        """
        расчет MTLD (Среднее значение прямого и обратного)

        min is 1.0
        higher is better
        """
        if not text or not text.strip():
            return 0.0

        lang = detect_main_language(text)
        tokens = nltk_helper.word_tokenize(text, lang)

        stopwords = self.stopwords_zh if lang == "zh" else self.stopwords_en
        filtered_tokens = [word for word in tokens if word not in stopwords]
        filtered_tokens = [word for word in filtered_tokens if word.isalnum()]

        if not filtered_tokens:
            return 0

        # forward MTLD
        forward_factors = self._compute_factors(filtered_tokens, threshold)

        # backward MTLD
        backward_factors = self._compute_factors(filtered_tokens[::-1], threshold)

        # average forward + backward
        return (forward_factors + backward_factors) / 2

    @staticmethod
    def _compute_factors(tokens: list, threshold: float) -> float:
        factors = 0
        current_segment = []
        unique_words = set()

        for token in tokens:
            current_segment.append(token)
            unique_words.add(token)
            ttr = len(unique_words) / len(current_segment)

            if ttr <= threshold:
                factors += 1
                current_segment = []
                unique_words = set()

        # last fragment
        if current_segment:
            ttr = len(unique_words) / len(current_segment)
            if ttr <= threshold:
                factors += 1
            else:
                factors += 1 - (ttr - threshold) / (1 - threshold)

        return len(tokens) / factors if factors > 0 else len(tokens)
