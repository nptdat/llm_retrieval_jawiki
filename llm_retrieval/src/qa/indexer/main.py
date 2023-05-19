import logging
from pathlib import Path
from time import time

from qa.vectorizer.model_vectorizer import ModelVectorizer
from qa.loader import JaWikiLoader
from qa.datastore import QdrantDataStore
from qa.schemas import QdrantConfig, IndexerConfig
from tqdm import tqdm


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    level="INFO",
    format="%(asctime)-15s %(module)s %(levelname)s - %(message)s",
)


def start():
    print("Start indexing jawiki...")
    start_time = time()

    config_path = Path("../config")

    cfg = IndexerConfig.load(config_path / "indexer.yml")
    cfg_qdrant = QdrantConfig.load(config_path / "qdrant.yml")
    logger.info(f"Indexer configuration: {cfg}")

    datastore = QdrantDataStore(cfg_qdrant, force_recreation=False)
    loader = JaWikiLoader(cfg.data_filepath, cfg.loader_debug_filepath)
    vectorizer = ModelVectorizer(
        cfg.vectorizer_base_model,
        cfg.vectorizer_base_class,
        batch_size=cfg.vectorizer_batch_size,
        device=cfg.vectorizer_device
    )

    cnt = 0
    with tqdm(total=cfg.limit_chunk_count) as pbar:
        for chunk_batch in loader(
            batch_size=cfg.loader_batch_size,
            max_len=cfg.max_chunk_len,
            limit=cfg.limit_chunk_count
        ):
            cnt += len(chunk_batch)
            if cnt < cfg.skip_count:
                pbar.update(cfg.loader_batch_size)
                continue

            texts = [chunk.text for chunk in chunk_batch]
            embeddings = vectorizer(texts)
            for idx, chunk in enumerate(chunk_batch):
                chunk.embedding = embeddings[idx].tolist()
            datastore.upsert(chunk_batch)

            pbar.update(cfg.loader_batch_size)

    logger.info(f"Finished in {time() - start_time:.4f} (secs)")
