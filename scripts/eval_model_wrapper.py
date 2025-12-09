"""DeepEval model wrapper for local vLLM/llama.cpp servers."""

from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI


class LocalServerLLM(DeepEvalBaseLLM):
    """Wrapper for OpenAI-compatible local servers (vLLM, llama.cpp).

    Implements the DeepEvalBaseLLM interface to allow running IFEval and
    HumanEval benchmarks against locally-hosted model servers.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "local-model",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=base_url, api_key="not-needed")

    def load_model(self):
        """No-op since server is already running."""
        pass

    def generate(self, prompt: str) -> str:
        """Generate a single response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        """Async generation (falls back to sync)."""
        return self.generate(prompt)

    def generate_samples(
        self, prompt: str, n: int, temperature: float
    ) -> list[str]:
        """Generate multiple samples for HumanEval pass@k calculation.

        Args:
            prompt: The code generation prompt.
            n: Number of samples to generate.
            temperature: Sampling temperature (higher = more diverse).

        Returns:
            List of n code completions.
        """
        completions = []
        for _ in range(n):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=self.max_tokens,
            )
            completions.append(response.choices[0].message.content)
        return completions

    def get_model_name(self) -> str:
        return self.model_name
