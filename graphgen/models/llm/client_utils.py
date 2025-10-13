from typing import Optional
import os
import tiktoken
from transformers import AutoTokenizer
import asyncio
import atexit
import hashlib
import os
import pickle
from typing import Any
import sys

def unpickle_data(file_path: str) -> Any:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def pickle_data(data: Any, file_path: str) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

class ApiCostTracker:
    def __init__(self, model_name: Optional[str] = None, 
        input_price: Optional[float] = None, 
        output_price: Optional[float] = None):        

        
        self.model_name = model_name 
        if self.model_name is None:
            raise ValueError("Model name must be provided or present in the session file.")
        self.tokenizer = self._get_tokenizer(self.model_name)
        self.input_price = input_price
        self.output_price = output_price

        self.num_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_input_cost = 0.0
        self.total_output_cost = 0.0
        self.total_cost = 0.0
        self.avg_input_cost = 0.0
        self.avg_output_cost = 0.0
        self.avg_cost = 0.0

        self._lock = asyncio.Lock()
    
    def _get_tokenizer(self, model_name: str):
        try:
            encoding = tiktoken.encoding_for_model(model_name)
            return lambda text: len(encoding.encode(text))
        except KeyError:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return lambda text: len(tokenizer.encode(text))


    async def track_cost(self, input_text: str, output_text: str):
        async with self._lock:
            # input_tokens = self.tokenizer(input_text) if input_text else 0 
            # output_tokens = self.tokenizer(output_text) if output_text else 0
            input_tokens=0
            output_tokens=0

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            self.total_input_cost = self.input_price * self.total_input_tokens / 1_000_000
            self.total_output_cost = self.output_price * self.total_output_tokens / 1_000_000
            self.total_cost = self.total_input_cost + self.total_output_cost

            self.num_requests += 1
            
            self.avg_input_cost = self.total_input_cost / self.num_requests
            self.avg_output_cost = self.total_output_cost / self.num_requests
            self.avg_cost = self.total_cost / self.num_requests


    def get_summary(self):
        return {
            "model_name": self.model_name,
            "input_price" : self.input_price,
            "output_price" : self.output_price,
            "total_input_tokens" : self.total_input_tokens,
            "total_output_tokens" : self.total_output_tokens,
            "total_input_cost" : self.total_input_cost,
            "total_output_cost" : self.total_output_cost,
            "total_cost" : self.total_cost,
            "num_requests" : self.num_requests,
            "avg_input_cost" : self.avg_input_cost,
            "avg_output_cost" : self.avg_output_cost,
            "avg_cost" : self.avg_cost,
        }


class CacheManager:
    def __init__(
        self,
        cache_file: str,
        hash_before_cache: bool = False,
    ) -> None:
        cache_dir = os.path.dirname(cache_file)
        if not os.path.exists(cache_dir):
            raise ValueError(
                f"Cache directory '{cache_dir}' does not exist. Please ensure "
                f"the directory exists before attempting to access '{cache_file}'."
            )

        self.cache_file = cache_file
        self.cache = (
            unpickle_data(self.cache_file) if os.path.exists(self.cache_file) else {}
        )
        self.hash_before_cache = hash_before_cache
        self._lock = asyncio.Lock()
        atexit.register(self.update_cache)

    def _prepare_key(self, primary_key: str) -> str:
        return (
            hashlib.sha256(primary_key.encode("utf-8")).hexdigest()
            if self.hash_before_cache
            else primary_key
        )

    async def __len__(self) -> int:
        async with self._lock:
            return len(self.cache)

    async def __getitem__(self, primary_key: str) -> Any:
        async with self._lock:
            return self.cache[self._prepare_key(primary_key)]

    async def __setitem__(self, primary_key: str, value: Any) -> None:
        async with self._lock:
            self.cache[self._prepare_key(primary_key)] = value

    async def __contains__(self, primary_key: str) -> bool:
        async with self._lock:
            return self._prepare_key(primary_key) in self.cache

    async def get(self, primary_key: str) -> Any:
        async with self._lock:
            return self.cache.get(self._prepare_key(primary_key))

    def update_cache(
        self,
    ):
        try:
            tmp_path = self.cache_file + ".tmp"
            pickle_data(self.cache, tmp_path)
            os.replace(tmp_path, self.cache_file)  # atomic write
        except Exception as e:
            print(f"Warning: failed to save KV cache: {e}", file=sys.stderr)
