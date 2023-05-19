from typing import Union
from pathlib import Path

from pydantic import BaseModel, validator
import yaml


class ConfigLoader(BaseModel):
    _CONFIG_FILENAME: str = "config.yml"

    @classmethod
    def load(cls, config_path: Path) -> BaseModel:
        with open(config_path / Path(cls._CONFIG_FILENAME)) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        return cls(**config)


class QdrantConfig(ConfigLoader):
    _CONFIG_FILENAME: str = "qdrant.yml"

    url: str
    http_port: int
    grpc_port: int
    prefer_grpc: bool = False
    collection_name: str
    api_key: Union[str, None] = None
    vector_size: int
    distance: str = "Cosine" # Any of "Cosine" / "Euclid" / "Dot". Distance function to measure
    timeout: int = 10


class IndexerConfig(ConfigLoader):
    _CONFIG_FILENAME: str = "indexer.yml"

    max_chunk_len: int
    vectorizer_batch_size: int
    loader_batch_size: int
    loader_debug_filepath: str
    limit_chunk_count: int = 10**8
    vectorizer_base_model: str
    vectorizer_base_class: str
    data_filepath: str
    vectorizer_device: str = "cuda"
    skip_count: int = 0

    @validator("limit_chunk_count")
    def type_cast(cls, v: float) -> Union[float, int]:
        if v == 0:
            return 10**8
        else:
            return v


class OpenAIConfig(ConfigLoader):
    _CONFIG_FILENAME: str = "openai.yml"

    api_key: str


class QAConfig(ConfigLoader):
    _CONFIG_FILENAME: str = "qa.yml"

    bearer_token: str
