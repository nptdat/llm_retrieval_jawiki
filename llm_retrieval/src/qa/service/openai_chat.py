from time import time
from logging import getLogger

import openai


logger = getLogger(__name__)

class OpenAIChat:
    MODEL_NAME = "gpt-3.5-turbo"

    def __init__(self, openai_api_key: str):
        openai.api_key = openai_api_key

    def complete(self, user_content: str) -> str:
        openai.Model.list()  # dummy call to refresh openai connection

        messages = [
            {"role": "system", "content": "貴方が役に立つアシスタントである。"},
            {"role": "user", "content": user_content}
        ]

        start_time = time()
        response = openai.ChatCompletion.create(
            model=self.MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0.2,
            presence_penalty=0.0,
            stream=False
        )

        logger.info(f"OpenAI API took {time()-start_time} (secs)")

        return response["choices"][0]["message"]["content"]
