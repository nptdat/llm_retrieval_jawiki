# Copied with changes from https://github.com/openai/chatgpt-retrieval-plugin/blob/main/datastore/datastore.py
from abc import ABC, abstractmethod
from typing import List, Union
import uuid

from qa.schemas import Chunk


class DataStore(ABC):
    UUID_NAMESPACE = uuid.UUID("1f026d71-cfa1-4945-9954-fcfd3f8dc4b0")

    def map_id(self, chunk_id: Union[int, str]) -> str:
        """
        Map a chunk_id to uuid space.
        If the chunk_id is unique, this function ensures that the generated
        string is also unique in the uuid space.
        """
        return uuid.uuid5(self.UUID_NAMESPACE, str(chunk_id)).hex

    def upsert(
        self, chunks: List[Chunk]
    ) -> List[str]:
        """
        First deletes all the existing vectors with the id (if necessary,
        depends on the vector db), then inserts the new ones.
        Return a list of ids.
        """
        self.delete(
            ids=[self.map_id(chunk.id) for chunk in chunks]
        )
        return self._upsert(chunks)

    @abstractmethod
    async def _upsert(self, chunks: List[Chunk]) -> List[str]:
        """
        Inserts chunks into the database.
        Return a list of mapped ids of chunk_ids.
        """
        raise NotImplementedError

    @abstractmethod
    def query(self, query_embedding: List[float], top_n: int=10) -> List[Chunk]:
        raise NotImplementedError

    @abstractmethod
    async def delete(
        self,
        ids: List[str]
    ) -> bool:
        raise NotImplementedError
