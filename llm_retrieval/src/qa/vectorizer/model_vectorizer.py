from typing import List

import numpy as np

from .base import BaseVectorizer
from qa.vectorizer.internal_model import *


class ModelVectorizer(BaseVectorizer):
    def __init__(
        self,
        model_name_or_path: str,
        model_class_name: str,
        batch_size: int,
        device: str
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.model_class_name = model_class_name
        self.batch_size = batch_size
        self.device = device
        self._load_model()

    def _load_model(self):
        class_obj = eval(self.model_class_name)
        self._model = class_obj(self.model_name_or_path, self.device)

    def __call__(self, texts: List[str]) -> np.array:
        embeddings = self._model.encode(texts, self.batch_size)
        return embeddings.detach().numpy()
