import hashlib
import json
from collections.abc import Awaitable
from typing import Any

from loguru import logger
from openai import OpenAI
from redis import Redis

from src.cfg.config import get_openai_api_key, get_redis_host, get_redis_port, get_redis_pwd


class RedisLLMCache:
    def __init__(self, host: str, port: int, password: str, ttl: int = 3600) -> None:
        self.client = Redis(host=host, port=port, password=password, decode_responses=True)
        self.ttl = ttl

    def _make_key(self, model: str, messages: list[dict], temperature: float) -> str:
        data = json.dumps(
            {"model": model, "messages": messages, "temperature": temperature},
            ensure_ascii=False,
            sort_keys=True,
        )
        return f"llm:{hashlib.sha256(data.encode()).hexdigest()}"

    def get(self, model: str, messages: list[dict], temperature: float) -> Awaitable[Any] | None | Any:
        key = self._make_key(model, messages, temperature)
        value = self.client.get(key)
        if value is not None:
            logger.debug("Cache hit: {}", key)
        else:
            logger.debug("Cache miss: {}", key)
        return value

    def set(self, model: str, messages: list[dict], temperature: float, response: str) -> None:
        key = self._make_key(model, messages, temperature)
        self.client.setex(key, self.ttl, response)
        logger.debug("Cache with TTL={}s: {}", self.ttl, key)

    def stats(self) -> dict[str, str | Awaitable[Any] | Any]:
        info = self.client.info("stats")
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": f"{hits / total * 100:.1f}%" if total else "N/A",
            "keys": self.client.dbsize(),
        }


def build_client() -> OpenAI:
    client = OpenAI(api_key=get_openai_api_key())
    return client


def chat_with_cache(
    client: Any,
    messages: list[dict],
    temperature: float = 0,
    model: str = "gpt-4o-mini",
    cache: Any | None = None,
) -> str:
    if cache:
        cached = cache.get(model, messages, temperature)
        if cached:
            logger.info("Ответ из кеша")
            return cached

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=50,
    )
    answer = response.choices[0].message.content

    if cache and temperature == 0:
        cache.set(model, messages, temperature, answer)
        logger.info("Сохранено в кеш (tokens: {:d})", response.usage.total_tokens)

    return answer


def print_cache_report(cache: Any | None = None, avg_tokens: int = 500) -> None:
    stats = cache.stats()
    cost_per_token = 2.50 / 1_000_000

    logger.info("=== Cache Report ===")
    logger.info("Keys in cache: {}", stats["keys"])
    logger.info("Hits:          {}", stats["hits"])
    logger.info("Misses:        {}", stats["misses"])
    logger.info("Hit rate:      {}", stats["hit_rate"])

    saved = stats["hits"] * avg_tokens * cost_per_token
    logger.info("Estimated savings: ${:.4f}", saved)


def run_redis_cache(messages: list[dict[str, str]]) -> None:
    client = build_client()
    cache = RedisLLMCache(host=get_redis_host(), port=get_redis_port(), password=get_redis_pwd(), ttl=3600)

    answer1 = chat_with_cache(client, messages, cache=cache)
    logger.info(f"Ответ: {answer1}")

    answer2 = chat_with_cache(client, messages, cache=cache)
    logger.info(f"Ответ: {answer2}")

    print_cache_report(cache)
