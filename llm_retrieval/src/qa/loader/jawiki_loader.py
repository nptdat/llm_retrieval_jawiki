import gzip
import json
import re
from tqdm import tqdm
from typing import Generator, Optional
import unicodedata
from pathlib import Path

from tqdm import tqdm
from .base import BaseLoader
from qa.schemas import Document


# Copied from: https://github.com/cl-tohoku/bert-japanese/blob/main/make_corpus_wiki.py#L54
def preprocess_text(text: str, title: Optional[str]=None) -> str:
    text = unicodedata.normalize("NFKC", text)

    # remove invisible characters
    text = "".join(c for c in text if c.isprintable())

    # remove templates
    text = re.sub(r"\[\d+?\]", "", text)
    text = re.sub(r"\[要.+?\]", "", text)
    text = re.sub(r"\{\{+[^{}]+?\}\}+", "", text)

    # remove navigation
    if title is not None:
        text = re.sub(r"^.+? \> " + re.escape(title), "", text)

    # remove footnotes
    text = re.sub(r" \^ .+", "", text)
    # remove annotations
    text = re.sub(r"\[(要出典|リンク切れ|.+?\?)\]", "", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


class JaWikiLoader(BaseLoader):
    def __init__(self, filepath: str, debug_filepath: Optional[str]=None) -> None:
        self.filepath = filepath
        self.debug_filepath = debug_filepath
        # self._reset_debug_file()

    def _reset_debug_file(self):
        filepath = Path(self.debug_filepath)
        if filepath.exists():
            filepath.unlink()

    def load_doc(self) -> Generator[Document, None, None]:
        with gzip.open(self.filepath, "rb") as fin:
            for _, line in tqdm(enumerate(fin)):
                dict_data = json.loads(line)
                page_id = dict_data.get("page_id")
                text = dict_data.get("text")
                if not text:
                    continue
                title = dict_data.get("title")
                text = preprocess_text(text, title=title)
                doc = Document(
                    id=str(page_id),
                    length=len(text),
                    title=title,
                    text=text
                )
                yield doc

                if self.debug_filepath:
                    with open(self.debug_filepath, "a") as f:
                        f.write(title + "\n")
