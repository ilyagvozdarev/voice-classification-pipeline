from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # Multi-label по умолчанию: go_emotions (28 меток эмоций, sigmoid).
    bert_model_name: str = "SamLowe/roberta-base-go_emotions"
    # Порог отсечения (sigmoid).
    bert_threshold: float = 0.3
    log_level: str = "INFO"


settings = Settings()
