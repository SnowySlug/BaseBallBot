"""Application configuration via Pydantic Settings."""

from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///data/db/bbbot.db"
    echo: bool = False


class MLBApiConfig(BaseModel):
    base_url: str = "https://statsapi.mlb.com/api/v1"
    timeout: int = 30
    max_retries: int = 3


class OddsApiConfig(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.the-odds-api.com/v4"
    regions: str = "us"
    markets: str = "h2h,totals,spreads"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    db: DatabaseConfig = DatabaseConfig()
    mlb_api: MLBApiConfig = MLBApiConfig()
    odds_api: OddsApiConfig = OddsApiConfig()

    weather_api_key: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""

    model_dir: Path = Path("data/models")
    cache_dir: Path = Path("data/cache")
    log_level: str = "INFO"
    season: int = 2026


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
