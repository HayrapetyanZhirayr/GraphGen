import math
import logging
from typing import Any, Dict, List, Optional

import openai
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from graphgen.bases.base_llm_client import BaseLLMClient
from graphgen.bases.datatypes import Token
from graphgen.models.llm.limitter import RPM, TPM
from .client_utils import ApiCostTracker, CacheManager
from graphgen.utils import logger
import json
CALL_LOG_INTERVAL = 10

def get_top_response_tokens(response: openai.ChatCompletion) -> List[Token]:
    token_logprobs = response.choices[0].logprobs.content
    tokens = []
    for token_prob in token_logprobs:
        prob = math.exp(token_prob.logprob)
        candidate_tokens = [
            Token(t.token, math.exp(t.logprob)) for t in token_prob.top_logprobs
        ]
        token = Token(token_prob.token, prob, top_candidates=candidate_tokens)
        tokens.append(token)
    return tokens


class OpenAIClient(BaseLLMClient):
    def __init__(
        self,
        *,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        json_mode: bool = False,
        seed: Optional[int] = None,
        topk_per_token: int = 5,  # number of topk tokens to generate for each token
        request_limit: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.json_mode = json_mode
        self.seed = seed
        self.topk_per_token = topk_per_token

        self.token_usage: list = []
        self.request_limit = request_limit
        self.rpm = RPM(rpm=6_000)
        self.tpm = TPM(tpm=2_500_000)

        self.api_cost_tracker = ApiCostTracker("Qwen/Qwen2.5-72B", 0.13, 0.40)
        self.cache_manager = CacheManager("/workspace/hayrapetyan/GraphGen/openai_cache.pkl")

        self.__post_init__()

    def __post_init__(self):
        assert self.api_key is not None, "Please provide api key to access openai api."
        self.client = AsyncOpenAI(
            api_key=self.api_key or "dummy", base_url=self.base_url
        )

    def _pre_generate(self, text: str, history: List[str]) -> Dict:
        kwargs = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.seed:
            kwargs["seed"] = self.seed
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": text})

        if history:
            assert len(history) % 2 == 0, "History should have even number of elements."
            messages = history + messages

        kwargs["messages"] = messages
        return kwargs

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, APITimeoutError)
        ),
    )
    async def generate_topk_per_token(
        self,
        text: str,
        history: Optional[List[str]] = None,
        **extra: Any,
    ) -> List[Token]:
        kwargs = self._pre_generate(text, history)
        if self.topk_per_token > 0:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = self.topk_per_token

        # Limit max_tokens to 1 to avoid long completions
        kwargs["max_tokens"] = 1

        try:
            key = json.dumps({"model":self.model_name, **kwargs}, ensure_ascii=False, sort_keys=True)
            if await self.cache_manager.__contains__(key):
                completion = await self.cache_manager[key]
            else:
                completion = await self.client.chat.completions.create(  # pylint: disable=E1125
                    model=self.model_name, **kwargs
                )
                await self.cache_manager.__setitem__(key, completion)
            if getattr(completion, "error", None):
                logger.warning(
                    "OpenAIClient API returned an error: %s", completion.error
                )
                return ""
            await self.api_cost_tracker.track_cost(
                input_text="\n".join(el["content"] for el in kwargs["messages"]),
                output_text=completion.choices[0].message.content)
            if self.api_cost_tracker.num_requests % CALL_LOG_INTERVAL == 0:
                summary = self.api_cost_tracker.get_summary()
                logger.info(
                    "OpenAIClient %s summary after %d calls: total tokens: %d (input), %d (output), total cost: %.6f",
                    summary["model_name"],
                    summary["num_requests"],
                    summary["total_input_tokens"],
                    summary["total_output_tokens"],
                    summary["total_cost"],
                )
        except openai.BadRequestError as err:
            # Handle context overflow due to oversized max_tokens
            msg = str(err)
            if "max_tokens" in msg or "max_completion_tokens" in msg or "maximum context length" in msg:
                logging.warning(
                    "openai.BadRequestError due to context overflow in generate_topk_per_token; returning empty list. model=%s", 
                    self.model_name,
                )
                return []
            else:
                raise

        tokens = get_top_response_tokens(completion)

        return tokens

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, APITimeoutError)
        ),
    )
    async def generate_answer(
        self,
        text: str,
        history: Optional[List[str]] = None,
        **extra: Any,
    ) -> str:
        kwargs = self._pre_generate(text, history)

        prompt_tokens = 0
        for message in kwargs["messages"]:
            prompt_tokens += len(self.tokenizer.encode(message["content"]))
        estimated_tokens = prompt_tokens + kwargs["max_tokens"]

        if self.request_limit:
            await self.rpm.wait(silent=False)
            await self.tpm.wait(estimated_tokens, silent=False)

        try:
            key = json.dumps({"model":self.model_name, **kwargs}, ensure_ascii=False, sort_keys=True)
            if await self.cache_manager.__contains__(key):
                completion = await self.cache_manager[key]
            else:
                completion = await self.client.chat.completions.create(  # pylint: disable=E1125
                    model=self.model_name, **kwargs
                )
                # тут нужно кэшить только если finish_reasin="stop"
                await self.cache_manager.__setitem__(key, completion)
            if getattr(completion, "error", None):
                logger.warning(
                    "OpenAIClient API returned an error: %s", completion.error
                )
                return ""
            await self.api_cost_tracker.track_cost(
                input_text="\n".join(el["content"] for el in kwargs["messages"]),
                output_text=completion.choices[0].message.content)
            if self.api_cost_tracker.num_requests % CALL_LOG_INTERVAL == 0:
                summary = self.api_cost_tracker.get_summary()
                logger.info(
                    "OpenAIClient %s summary after %d calls: total tokens: %d (input), %d (output), total cost: %.6f",
                    summary["model_name"],
                    summary["num_requests"],
                    summary["total_input_tokens"],
                    summary["total_output_tokens"],
                    summary["total_cost"],
                )
        except openai.BadRequestError as err:
            # Handle context overflow due to oversized max_tokens
            msg = str(err)
            if "max_tokens" in msg or "max_completion_tokens" in msg or "maximum context length" in msg:
                logging.warning(
                    "openai.BadRequestError due to context overflow in generate_answer; returning empty string. model=%s",
                    self.model_name,
                )
                return ""
            else:
                raise

        if hasattr(completion, "usage"):
            self.token_usage.append(
                {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens,
                }
            )
        return self.filter_think_tags(completion.choices[0].message.content)

    async def generate_inputs_prob(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        """Generate probabilities for each token in the input."""
        raise NotImplementedError
