from pathlib import Path
from logging import getLogger

from qa.datastore import QdrantDataStore
from qa.schemas import QdrantConfig, IndexerConfig, OpenAIConfig
from qa.vectorizer.model_vectorizer import ModelVectorizer
from .openai_qa import OpenAIQA


logger = getLogger(__name__)


class Answerer:
    def __init__(self, config_path: Path):
        cfg = IndexerConfig.load(config_path)
        cfg_openai = OpenAIConfig.load(config_path)
        cfg_qdrant = QdrantConfig.load(config_path)

        self.datastore = QdrantDataStore(cfg_qdrant, force_recreation=False)
        self.vectorizer = ModelVectorizer(
            cfg.vectorizer_base_model,
            cfg.vectorizer_base_class,
            batch_size=cfg.vectorizer_batch_size,
            device=cfg.vectorizer_device
        )
        self.qa = OpenAIQA(cfg_openai.api_key)

    def answer(self, query: str) -> str:
        query_embeddings = self.vectorizer([query])[0]
        logger.info(f"{query_embeddings=}")
        chunks = self.datastore.query(query_embeddings, top_n=5)
        for chunk in chunks:
            logger.info(f"{chunk=}\n")
        return self.qa(
            question=query,
            references=chunks[:2]
        )
