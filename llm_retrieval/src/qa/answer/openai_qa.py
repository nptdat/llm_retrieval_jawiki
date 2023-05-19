from typing import List
from logging import getLogger

from qa.service.openai_chat import OpenAIChat


logger = getLogger(__name__)


class OpenAIQA(OpenAIChat):
    PROMPT_TPL = """
    以下の各参考テキストを基づいて質問を回答してください。
    {references}
    {question}
    """
    REFERENCE_TPL = "参考テキスト{index}: {reference}"

    def answer(self, question: str, references: List[str]) -> str:
        logger.info(f"{references=}")
        reference_text = "".join([
            self.REFERENCE_TPL.format(index=idx, reference=reference)
            for idx, reference in enumerate(references)
        ])
        prompt = self.PROMPT_TPL.format(
            references=reference_text,
            question=question
        )
        logger.info(f"{logger=}")
        answer = self.complete(user_content=prompt)
        return answer

    def __call__(self, question: str, references: List[str]) -> str:
        return self.answer(question, references)
