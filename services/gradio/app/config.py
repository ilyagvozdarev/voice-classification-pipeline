from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    master_url: str = "http://master:8000"
    request_timeout_s: int = 180
    log_level: str = "INFO"


settings = Settings()
