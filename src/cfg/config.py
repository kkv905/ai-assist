import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APIStatusError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()


def get_env_required(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Не найдена переменная {key}. Проверь файл .env и загрузку через load_dotenv().")
    return value


def get_openai_api_key() -> str:
    """Получаем ключ для OpenAI"""
    return get_env_required("OPENAI_API_KEY")


def get_openrouter_api_key() -> str:
    """получаем ключ для OpenRouter"""
    return get_env_required("OPENROUTER_API_KEY")


def get_hugging_face_token() -> str:
    """Получаем токен для Hugging Face"""
    return get_env_required("HF_TOKEN")

def get_redis_pwd() -> str:
    """Получаем пароль от Redis"""
    return get_env_required("REDIS_PWD")

def get_redis_host() -> str:
    """Получаем host для Redis"""
    return get_env_required("REDIS_HOST")

def get_redis_port() -> int:
    """Получаем порт Redis"""
    return int(get_env_required("REDIS_PORT"))


def call_llm_with_retry(client: Any, model: str, messages: list[dict]) -> Any:
    """Вызов LLM с автоматическим retry через tenacity.
    Повторяет запрос при:
    - RateLimitError (429) — до 5 раз с экспоненциальной задержкой
    - APIStatusError с кодом 5xx — серверные ошибки
    """
    # Функция-предикат: повторяем на 429 и 5xx
    def should_retry(error: BaseException) -> bool:
        if isinstance(error, RateLimitError):
            return True
        if isinstance(error, APIStatusError) and error.status_code >= 500:
            return True
        return False

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((RateLimitError, APIStatusError)),
    )
    def _call() -> Any:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )

    return _call()

def call_with_fallback(
    primary_client: Any,
    primary_model: str,
    fallback_client: Any | None,
    fallback_model: str | None,
    messages: list[dict],
) -> Any:
    """Пробует основного провайдера, при неудаче — fallback.
    Возвращает stream от первого успешного провайдера.
    """
    try:
        return call_llm_with_retry(primary_client, primary_model, messages), primary_model
    except (RateLimitError, APIError, APIConnectionError) as e:
        print(f"  [Основной провайдер недоступен: {e}]")

        if fallback_client and fallback_model:
            print(f"  [Переключаюсь на fallback: {fallback_model}]")
            try:
                return call_llm_with_retry(fallback_client, fallback_model, messages), fallback_model
            except Exception as fallback_err:
                raise RuntimeError(
                    f"Оба провайдера недоступны. Fallback: {fallback_err}"
                ) from fallback_err
        raise


def build_client(provider: str, model: str) -> tuple[Any, str]:
    """Создаёт OpenAI-совместимый клиент для указанного провайдера и модели."""
    providers_path = Path(__file__).resolve().parents[2] / "appsetting" / "providers.json"
    try:
        with providers_path.open("r", encoding="utf-8") as f:
            providers = json.load(f)
    except FileNotFoundError as exc:
        raise SystemExit(f"Файл провайдеров не найден: {providers_path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Некорректный JSON в файле провайдеров: {providers_path}") from exc

    if provider not in providers:
        available = ", ".join(providers)
        raise SystemExit(f"Неизвестный провайдер '{provider}'. Доступно: '{available}'")

    cfg = providers[provider]
    cfg["model"] = model
    env_key = cfg.get("env_key")
    api_key = get_env_required(env_key) if env_key else cfg.get("api_key")
    if not api_key:
        raise SystemExit(f"Для провайдера '{provider}' не настроен API-ключ.")

    client = OpenAI(base_url=cfg["base_url"], api_key=api_key)
    return client, cfg["model"]
