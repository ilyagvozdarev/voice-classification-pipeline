from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    bert_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    log_level: str = "INFO"


settings = Settings()
