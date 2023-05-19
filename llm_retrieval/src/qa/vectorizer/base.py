from abc import ABC, abstractmethod
from typing import List

import numpy as np
from qa.schemas import Chunk


class BaseVectorizer(ABC):
    @abstractmethod
    def _load_model(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, chunks: List[Chunk]) -> np.array:
        raise NotImplementedError
