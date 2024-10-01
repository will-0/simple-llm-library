from .. import LLM, LLMResponse, LLMTool
import os
from typing import List
import json
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY")
)

class GPT4o(LLM):

    def __init__(self, tools: List[LLMTool]):
        self.tools = tools
        pass

    @property
    def can_use_tools(self):
        return True

    def call(self, prompt: str, force_tools: bool) -> LLMResponse:

        # Create the tools object
        tools=[
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.call_spec.model_json_schema(),
                },
            }
            for tool in self.tools
        ]

        # If forcing tool use, ensure only one tool is supplied
        if force_tools and len(self.tools) < 1:
            raise Exception("If forcing tools, a tool must be supplied")
        if force_tools and len(self.tools) > 1:
            raise Exception("If forcing tools, only one tool can be supplied")

        # Create the chat completion
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": self.tools[0].name},
            } if force_tools else None,
            model="gpt-4o",
        )

        # Assert length of choices is 1
        assert len(chat_completion.choices) == 1
        choice = chat_completion.choices[0]

        # Check if the tool was called
        did_call_tool = len(choice.message.tool_calls) > 0
        if did_call_tool:
            tool_call = choice.message.tool_calls[0]
            return LLMResponse(
                content=json.loads(tool_call.function.arguments),
                content_type='tool_call'
            )
        else:
            return LLMResponse(
                content=chat_completion.choices[0].message.content,
                content_type='text'
            )


    