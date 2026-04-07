from src.model_client import (
    BaseModelClient,
    ModelConfig,
    MalformedResponseError,
    APIMessage,
)
import os
import time
from openai import (
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)


class OpenRouterClient(BaseModelClient):
    """Client for interacting directly with OpenRouter using the OpenAI SDK."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the OpenRouter client with specific API parameters."""
        super().__init__(config)
        api_key = self.config.api_key or os.environ.get("OPENROUTER_API_KEY")
        api_base = self.config.api_base or "https://openrouter.ai/api/v1"

        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key,
        )

    def generate_response(self, messages: list[APIMessage]) -> str | None:
        """Generate a response using the OpenAI SDK with resilience and backoff."""
        attempt = 0
        current_backoff = self.config.initial_backoff

        if self.config.pacing_delay > 0:
            time.sleep(self.config.pacing_delay)

        while attempt <= self.config.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,  # type: ignore
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                    extra_body={
                        "reasoning": {
                            "enabled": True,
                            "max_tokens": self.config.max_tokens - 1000,
                            "exclude": True,
                        },
                    },
                )
                return self._unpack_response(response)

            except RateLimitError as e:
                logger.warning(
                    f"Rate Limit Hit! The API is demanding a pause. Details: {e}",
                )
                long_pause = 10
                logger.info(f"Sleeping for {long_pause} seconds to let quotas reset...")
                time.sleep(long_pause)
                attempt += 1

            except TimeoutError as e:
                logger.error(f"Timeout error for model {self.config.model_name}: {e}")
                return None

            except (ConnectionError, InternalServerError) as e:
                logger.warning(
                    f"Network error (Attempt {attempt + 1}/"
                    f"{self.config.max_retries + 1}): {e}",
                )
                time.sleep(current_backoff)
                current_backoff *= self.config.backoff_factor
                attempt += 1

            except MalformedResponseError as e:
                logger.error(
                    f"Malformed response error for model {self.config.model_name}: {e}",
                )
                return None

            except Exception as e:
                logger.error(
                    f"Unexpected API error for model {self.config.model_name}: {e}",
                )
                return None

        return None
