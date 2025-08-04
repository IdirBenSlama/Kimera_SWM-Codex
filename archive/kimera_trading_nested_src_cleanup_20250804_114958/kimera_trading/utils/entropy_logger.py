import logging

logger = logging.getLogger(__name__)


class EntropyLogger:
    """A logger that is aware of the system's entropy."""

    def log(self, message, entropy):
        # Placeholder for entropy-aware logging
        logger.info(f"[Entropy: {entropy:.2f}] {message}")
