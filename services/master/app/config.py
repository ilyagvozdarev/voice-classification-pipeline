from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # Downstream services are reached by their compose service name.
    asr_url: str = "http://asr:8000"
    bert_url: str = "http://bert:8000"
    llm_url: str = "http://llm:8000"

    request_timeout_s: int = 120
    log_level: str = "INFO"


settings = Settings()
