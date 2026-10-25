from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    es_host: str = "elasticsearch"
    es_port: int = 9200

    otel_enabled: bool = True
    otel_service_name: str = "smart-search-api"
    otel_endpoint: str = "http://otel-collector:4317"

    @property
    def es_url(self) -> str:
        return f"http://{self.es_host}:{self.es_port}"


settings = Settings()
