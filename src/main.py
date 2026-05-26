import os
from collections import deque
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.cache.redisllmcache import RedisLLMCache
from src.cfg.config import build_client, call_with_fallback, get_redis_host, get_redis_port, get_redis_pwd
from src.cfg.logging import setup_logging


def load_system_prompt() -> str:
    """Загружает системный промпт из файла проекта."""
    prompt_path = Path(__file__).resolve().parents[1] / "appsetting" / "system-prompt.md"
    return prompt_path.read_text(encoding="utf-8").strip()


def classify_message(text: str) -> str:
    """Классифицирует сообщение пользователя: FAQ / тех. проблема / жалоба."""
    normalized = text.lower()
    if any(word in normalized for word in ("не работает", "ошибка", "сбой", "падает", "не открывается")):
        return "тех. проблема"
    if any(word in normalized for word in ("жалоба", "ужасно", "плохо", "недоволен", "безобразие")):
        return "жалоба"
    return "FAQ"


def is_dissatisfied(text: str) -> bool:
    """Определяет, что пользователь недоволен последним ответом помощника."""
    normalized = text.lower()
    markers = (
        "не помогло",
        "не сработало",
        "это не то",
        "не решил",
        "неправильно",
        "все еще не работает",
        "не устраивает",
        "не доволен",
        "бесполезно",
    )
    return any(marker in normalized for marker in markers)


def build_escalation_message() -> str:
    """Формирует сообщение эскалации с коротким номером обращения."""
    return f"Передаю вопрос специалисту. Номер обращения: {uuid4().hex[:8].upper()}."


def build_messages(system_prompt: str, history: deque[dict[str, str]], user_text: str) -> list[dict[str, str]]:
    """Формирует сообщения для модели: system + последние 10 сообщений + текущий запрос."""
    return [{"role": "system", "content": system_prompt}, *list(history), {"role": "user", "content": user_text}]


def build_cache_messages(system_prompt: str, user_text: str) -> list[dict[str, str]]:
    """Формирует стабильный ключ кеша только из системного промпта и вопроса пользователя."""
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]


def stream_response(stream: Any) -> str:
    """Считывает поток ответа модели и возвращает итоговый текст."""
    chunks: list[str] = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            print(delta, end="", flush=True)
            chunks.append(delta)
    print()
    return "".join(chunks).strip()


def main() -> None:
    setup_logging()

    primary_provider = os.getenv("PRIMARY_PROVIDER", "2")
    primary_model = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
    fallback_provider = os.getenv("FALLBACK_PROVIDER", "Ollama (локальный)")
    fallback_model = os.getenv("FALLBACK_MODEL", "gemma3:1b")

    primary_client, primary_model = build_client(primary_provider, primary_model)
    try:
        fallback_client, fallback_model = build_client(fallback_provider, fallback_model)
    except SystemExit:
        fallback_client, fallback_model = None, None

    cache = RedisLLMCache(host=get_redis_host(), port=get_redis_port(), password=get_redis_pwd(), ttl=3600)
    system_prompt = load_system_prompt()
    history: deque[dict[str, str]] = deque(maxlen=10)
    dissatisfaction_streak = 0

    print("CLI-помощник запущен. Команды: /clear, /stats, /quit")
    while True:
        user_text = input("Вы: ").strip()
        if not user_text:
            continue
        if user_text == "/quit":
            print("Завершение работы.")
            break
        if user_text == "/clear":
            history.clear()
            dissatisfaction_streak = 0
            print("История диалога очищена.")
            continue
        if user_text == "/stats":
            stats = cache.stats()
            print(f"Cache hits: {stats['hits']}")
            print(f"Cache misses: {stats['misses']}")
            print(f"Cache hit rate: {stats['hit_rate']}")
            print(f"Ключей в Redis: {stats['keys']}")
            continue

        print(f"[Класс сообщения: {classify_message(user_text)}]")

        if is_dissatisfied(user_text):
            dissatisfaction_streak += 1
        else:
            dissatisfaction_streak = 0
        if dissatisfaction_streak >= 3:
            print(build_escalation_message())
            dissatisfaction_streak = 0

        messages = build_messages(system_prompt, history, user_text)
        cache_messages = build_cache_messages(system_prompt, user_text)
        cached = cache.get(primary_model, cache_messages, temperature=0)
        if cached:
            print(f"Помощник: {cached}")
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": cached})
            continue

        print("Помощник: ", end="", flush=True)
        try:
            stream, _model_used = call_with_fallback(
                primary_client=primary_client,
                primary_model=primary_model,
                fallback_client=fallback_client,
                fallback_model=fallback_model,
                messages=messages,
            )
            answer = stream_response(stream)
            # Для повторных вопросов в одной сессии кешируем по стабильному ключу (без истории).
            cache.set(primary_model, cache_messages, temperature=0, response=answer)
        except Exception as exc:
            print(f"Ошибка при обращении к модели: {exc}")
            continue

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
