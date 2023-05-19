from abc import ABC, abstractmethod
from typing import Generator, List, Tuple

from qa.schemas import Document, Chunk

AVAILABLE_DELIMITERS = ["\n", "。"]


class BaseLoader(ABC):
    def _split_doc_to_chunk(self, doc: Document, max_len: int) -> List[Chunk]:
        if doc.length <= max_len:
            return [Chunk(
                id=doc.id,
                length=doc.length,
                doc_title=doc.title,
                text=doc.text
            )]
        else:
            for delim in AVAILABLE_DELIMITERS:
                if delim in doc.text:
                    break
            lines = doc.text.split(delim)

            chunks = []
            chunk_order = 0
            chunk_lines = []
            chunk_len = 0
            for _, line in enumerate(lines):
                if chunk_len + len(line) > max_len:
                    chunks.append(Chunk(
                        id=f"{doc.id}_{chunk_order}",
                        length=chunk_len - 1,
                        order_in_doc=chunk_order,
                        doc_title=doc.title,
                        text=delim.join(chunk_lines)
                    ))
                    chunk_order += 1
                    chunk_lines = []
                    chunk_len = 0

                chunk_lines.append(line)
                chunk_len += len(line) + 1

            chunks.append(Chunk(
                id=f"{doc.id}_{chunk_order}",
                length=chunk_len - 1,
                order_in_doc=chunk_order,
                doc_title=doc.title,
                text=delim.join(chunk_lines)
            ))

            num_chunks_in_doc = len(chunks)
            for chunk in chunks:
                chunk.num_chunks_in_doc = num_chunks_in_doc
            return chunks

    def __call__(
        self,
        batch_size: int,
        max_len: int,
        limit: int
    ) -> Generator[List[Chunk], None, None]:
        chunks = []
        cnt = 0
        for doc in self.load_doc():
            _chunks = self._split_doc_to_chunk(doc, max_len)
            chunks.extend(_chunks)
            if len(chunks) >= batch_size:
                yield chunks[:batch_size]
                chunks = chunks[batch_size:]
                cnt += batch_size
                if limit > 0 and cnt >= limit:
                    return None

    @abstractmethod
    def load_doc(self) -> Generator[Document, None, None]:
        raise NotImplementedError
