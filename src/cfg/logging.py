from pathlib import Path

from loguru import logger


def setup_logging() -> None:
    project_root = Path(__file__).resolve().parents[2]
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        logs_dir / "log_{time:YYYYMMDD}.txt",
        rotation="00:00",
        level="DEBUG",
        encoding="utf-8",
    )
