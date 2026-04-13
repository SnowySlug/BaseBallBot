"""Custom exception hierarchy."""


class BBBotError(Exception):
    """Base exception for all bbbot errors."""


class APIError(BBBotError):
    """Error communicating with an external API."""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


class DataNotFoundError(BBBotError):
    """Requested data does not exist."""


class ConfigurationError(BBBotError):
    """Invalid or missing configuration."""
