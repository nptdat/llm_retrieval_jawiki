from typing import List

from transformers import T5Tokenizer, T5Model
import torch
import numpy as np


class SentenceT5:
    """Class to use the [sentence-t5-base-ja-mean-token](https://huggingface.co/sonoisa/sentence-t5-base-ja-mean-tokens)
    """
    def __init__(self, model_name_or_path: str, device=None):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, is_fast=False)
        self.model = T5Model.from_pretrained(model_name_or_path).encoder
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, docs: List[str], batch_size=8):
        all_embeddings = []
        iterator = range(0, len(docs), batch_size)
        for batch_idx in iterator:
            batch = docs[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(
                batch,
                max_length=4096,
                padding="longest",
                truncation=True,
                return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings)
