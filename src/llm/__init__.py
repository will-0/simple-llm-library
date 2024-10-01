from abc import ABC, abstractmethod
from typing import Literal, Optional, List, Type
from pydantic import BaseModel

# TYPE DEFINITIONS

class LLMResponse(BaseModel):
    content: str | dict
    content_type: Literal["text", "tool_call"]

class LLMTool(BaseModel):
    name: str
    description: Optional[str]
    call_spec: Type[BaseModel]

class LLM(ABC):
    """An LLM that can be called with a specific prompt. May or may not have function calling."""

    @abstractmethod
    def call(self, prompt: str) -> LLMResponse:
        """Takes a prompt message and returns a str"""
        pass

    @property
    def can_use_tools(self) -> bool:
        """A property for checking whether that specific LLM can use tools"""
        pass

class LLMFactory:
    """A method for creating LLMs with a specific set of tools"""

    @staticmethod
    def create_LLM(spec: str, tools: Optional[List[LLMTool]]) -> LLM:

        match spec:
            case 'gpt4o':
                from .openai.gpt4o import GPT4o
                return GPT4o(tools)
            case _:
                raise Exception(f"Implementation not supported for spec string \"{spec}\"")