from dataclasses import dataclass
import logging

@dataclass
class LoggingParams:
    level: int = logging.INFO
    fmt: str = "[%(asctime)s] %(levelname)s in %(name)s: %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"

    def configure(self, name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(self.fmt, datefmt=self.datefmt))
            logger.addHandler(handler)
            logger.setLevel(self.level)
        return logger
