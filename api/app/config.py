from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    es_host: str = "elasticsearch"
    es_port: int = 9200

    @property
    def es_url(self) -> str:
        return f"http://{self.es_host}:{self.es_port}"


settings = Settings()