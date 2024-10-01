import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.llm import LLMFactory

gpt4o = LLMFactory.create_LLM('gpt4o', None)

res = gpt4o.call("Say 'Hello World!'")

print(res.content)