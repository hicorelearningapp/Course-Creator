#ai_client.py
from typing import List, Dict, Any
from config import Config

class AIClient:
    """Abstracts the AI client (OpenAI or Azure)."""

    def __init__(self, use_azure: bool = False):
        self.use_azure = use_azure
        if use_azure:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                api_key=Config.AZURE_OPENAI_KEY,
                api_version=Config.AZURE_OPENAI_API_VERSION,
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT
            )
            self.deployment = "gpt-4.1"  # Azure deployment name
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            self.deployment = "gpt-4.1"  # can change per topic

    def get_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1000) -> Any:
        """Wrapper for chat completions."""
        return self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
