from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    llm_model_name: str = "google/flan-t5-small"
    llm_max_new_tokens: int = 128
    log_level: str = "INFO"


settings = Settings()
