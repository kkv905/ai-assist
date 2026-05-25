import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Не найдена переменная окружения OPENAI_API_KEY. Проверь файл .env и загрузку через load_dotenv()."
        )
    return api_key


def get_openrouter_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Не найдена переменная OPENROUTER_API_KEY. Проверь файл .env и загрузку через load_dotenv().")
    return api_key


def get_hugging_face_token() -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Не найдена переменная HF_TOKEN. Проверь файл .env и загрузку через load_dotenv().")
    return token

def get_redis_pwd() -> str:
    pwd = os.getenv("REDIS_PWD")
    if not pwd:
        raise ValueError("Не найдена переменная REDIS_PWD. Проверь файл .env и загрузку через load_dotenv().")
    return pwd

def get_redis_host() -> str:
    host = os.getenv("REDIS_HOST")
    if not host:
        raise ValueError("Не найдена переменная REDIS_HOST. Проверь файл .env и загрузку через load_dotenv().")
    return host

def get_redis_port() -> int:
    port = os.getenv("REDIS_PORT")
    if not port:
        raise ValueError("Не найдена переменная REDIS_PORT. Проверь файл .env и загрузку через load_dotenv().")
    return int(port)


def get_client(provider: str, model: str) -> tuple[Any, str]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Установите зависимость openai: pip install openai") from exc

    configs = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key": get_openai_api_key(),
            "model": model,
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": get_openrouter_api_key(),
            "model": model,
        },
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "model": model,
        },
    }

    if provider not in configs:
        available = ", ".join(configs)
        raise SystemExit(f"Неизвестный провайдер '{provider}'. Доступно: '{available}'")

    cfg = configs[provider]
    if not cfg["api_key"]:
        raise SystemExit(f"Для провайдера '{provider}' не настроен API-ключ.")

    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    return client, cfg["model"]
