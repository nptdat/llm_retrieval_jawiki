# Copied with changes from https://github.com/openai/chatgpt-retrieval-plugin/blob/main/datastore/providers/qdrant_datastore.py
from typing import List

from grpc._channel import _InactiveRpcError
import qdrant_client
from qdrant_client.http.exceptions import UnexpectedResponse

from .datastore import DataStore
from qdrant_client.http import models as rest
from qa.schemas import Chunk, ChunkSearchResult, QdrantConfig


class QdrantDataStore(DataStore):
    def __init__(self, cfg: QdrantConfig, force_recreation: bool=False) -> None:
        self.collection_name = cfg.collection_name
        self.client = qdrant_client.QdrantClient(
            url=cfg.url,
            port=int(cfg.http_port),
            grpc_port=int(cfg.grpc_port),
            api_key=cfg.api_key,
            prefer_grpc=cfg.prefer_grpc,
            timeout=cfg.timeout,
        )

        # Set up the collection so the points might be inserted or queried
        self._set_up_collection(cfg.vector_size, cfg.distance, force_recreation)

    def _upsert(self, chunks: List[Chunk]) -> List[str]:
        """
        Takes in a list of chunks and inserts them into the database.
        Return a list of document ids.
        """
        ids = [self.map_id(chunk.id) for chunk in chunks]
        points = [
            rest.PointStruct(
                id=id_,
                vector=chunk.embedding,  # type: ignore
                payload={
                    "id": chunk.id,
                    "length": chunk.length,
                    "order_in_doc": chunk.order_in_doc,
                    "num_chunks_in_doc": chunk.num_chunks_in_doc,
                    "doc_title": chunk.doc_title,
                    "text": chunk.text,
                },
            )
            for id_, chunk in zip(ids, chunks)
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,  # type: ignore
            wait=True,
        )
        return ids

    def query(self, query_embedding: List[float], top_n: int=10) -> List[ChunkSearchResult]:
        """
        Takes in a query embedding and returns a list of top N chunks whose
        embedding vectors close to the given embedding.
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_n
        )
        return [
            ChunkSearchResult(**result.payload, score=result.score)
            for result in results
        ]

    def delete(
        self,
        ids: List[str]
    ) -> bool:
        """
        Removes vectors by ids from the datastore.
        Returns whether the operation was successful.
        """
        response = self.client.delete(
            collection_name=self.collection_name,
            points_selector=rest.PointIdsList(
                points=[self.map_id(id) for id in ids]
            )
        )
        return "COMPLETED" == response.status

    def _set_up_collection(
        self, vector_size: int, distance: str, force_recreation: bool=False
    ):
        distance = rest.Distance[distance.upper()]
        if force_recreation:
            self._recreate_collection(distance, vector_size)

        try:
            collection_info = self.client.get_collection(self.collection_name)
            current_distance = collection_info.config.params.vectors.distance  # type: ignore
            current_vector_size = collection_info.config.params.vectors.size  # type: ignore

            if current_distance != distance:
                raise ValueError(
                    f"Collection '{self.collection_name}' already exists in Qdrant, "
                    f"but it is configured with a similarity '{current_distance.name}'. "
                    f"If you want to use that collection, but with a different "
                    f"similarity, please set `recreate_collection=True` argument."
                )

            if current_vector_size != vector_size:
                raise ValueError(
                    f"Collection '{self.collection_name}' already exists in Qdrant, "
                    f"but it is configured with a vector size '{current_vector_size}'. "
                    f"If you want to use that collection, but with a different "
                    f"vector size, please set `recreate_collection=True` argument."
                )
        except (UnexpectedResponse, _InactiveRpcError):
            self._recreate_collection(distance, vector_size)

    def _recreate_collection(self, distance: rest.Distance, vector_size: int):
        self.client.recreate_collection(
            self.collection_name,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=distance,
            ),
        )
