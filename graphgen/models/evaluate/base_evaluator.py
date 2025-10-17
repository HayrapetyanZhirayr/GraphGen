import asyncio
from dataclasses import dataclass

from tqdm.asyncio import tqdm as tqdm_async

from graphgen.bases.datatypes import QAPair
from graphgen.utils import create_event_loop


@dataclass
class BaseEvaluator:
    max_concurrent: int = 100
    results: list[float] = None

    def evaluate(self, pairs: list[QAPair]) -> list[float]:
        """
        Evaluate the text and return a score.
        """
        return create_event_loop().run_until_complete(self.async_evaluate(pairs))

    async def async_evaluate(self, pairs: list[QAPair]) -> list[float]:
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def evaluate_with_semaphore(pair, index):
            async with semaphore:  # 获取Semaphore
                result = await self.evaluate_single(pair)
            return index, result

        results = [None]*len(pairs)
        for result in tqdm_async(
            asyncio.as_completed([evaluate_with_semaphore(pair, index) for index, pair in enumerate(pairs)]),
            total=len(pairs),
        ):
            index, result = await result
            results[index] = result
        return results

    async def evaluate_single(self, pair: QAPair) -> float:
        raise NotImplementedError()

    def get_average_score(self, pairs: list[QAPair]) -> float:
        """
        Get the average score of a batch of texts.
        """
        results = self.evaluate(pairs)
        self.results = results
        return sum(self.results) / len(pairs)

    def get_min_max_score(self, pairs: list[QAPair]) -> tuple[float, float]:
        """
        Get the min and max score of a batch of texts.
        """
        if self.results is None:
            self.get_average_score(pairs)
        return min(self.results), max(self.results)
