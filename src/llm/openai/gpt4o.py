from .. import LLM, LLMResponse, LLMTool
import os
from typing import List
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY")
)

class GPT4o(LLM):

    def __init__(self, tools: List[LLMTool]):
        pass

    @property
    def can_use_tools(self):
        return True

    def call(self, prompt: str) -> LLMResponse:

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
        )


        response = LLMResponse(
            content=chat_completion.choices[0].message.content,
            content_type='text'
        )

        return response


    