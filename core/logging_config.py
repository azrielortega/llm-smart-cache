import logging

from core.config import LOG_LEVEL

# Third-party libraries whose default INFO logging (HTTP requests, model
# downloads) is noise here - keep them at WARNING regardless of LOG_LEVEL.
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock")


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
