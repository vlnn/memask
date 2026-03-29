from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
DEFAULT_MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF"
    "/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
)
MEMASK_DIR = ".memask"


def models_dir(*, base: Path | None = None) -> Path:
    parent = base or Path.home() / MEMASK_DIR
    return parent / "models"


def model_path(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    base: Path | None = None,
) -> Path:
    return models_dir(base=base) / model_name


def model_status(*, base: Path | None = None) -> dict:
    path = model_path(base=base)
    downloaded = path.is_file()
    size_mb = path.stat().st_size / (1024 * 1024) if downloaded else 0.0
    return {
        "model_name": DEFAULT_MODEL_NAME,
        "path": str(path),
        "downloaded": downloaded,
        "size_mb": round(size_mb, 1),
        "url": DEFAULT_MODEL_URL,
    }


def download_model(
    *,
    url: str = DEFAULT_MODEL_URL,
    base: Path | None = None,
    progress_callback=None,
) -> Path:
    dest = model_path(base=base)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file():
        logger.info("model already exists at %s", dest)
        return dest

    import urllib.request

    logger.info("downloading model from %s", url)

    def _report(block_num, block_size, total_size):
        if progress_callback and total_size > 0:
            downloaded = block_num * block_size
            pct = min(100.0, downloaded / total_size * 100)
            progress_callback(pct, downloaded, total_size)

    urllib.request.urlretrieve(url, str(dest), reporthook=_report)
    logger.info("model saved to %s", dest)
    return dest
