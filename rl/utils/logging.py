import sys
import logging


def configure_logger(debug_loggers: list[str] | None = []):
    root_logger = logging.getLogger()
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.WARNING)

    if debug_loggers is not None:
        for name in debug_loggers:
            logging.getLogger(name).setLevel(logging.DEBUG)
