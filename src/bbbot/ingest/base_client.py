"""Abstract base HTTP client with retry logic and rate limiting."""

import time
from abc import ABC

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from bbbot.exceptions import APIError

log = structlog.get_logger()


class BaseClient(ABC):
    """Base class for all API clients with retry and rate limiting."""

    def __init__(self, base_url: str, timeout: int = 30, max_retries: int = 3,
                 rate_limit: float = 1.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval = 1.0 / rate_limit  # seconds between requests
        self._last_request_time = 0.0
        self._client = httpx.Client(timeout=timeout)

    def _rate_limit(self):
        """Enforce minimum interval between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.monotonic()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
    )
    def _get(self, path: str, params: dict | None = None) -> dict:
        """Make a GET request with retry and rate limiting."""
        self._rate_limit()
        url = f"{self.base_url}/{path.lstrip('/')}"
        log.debug("api_request", url=url, params=params)

        response = self._client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
