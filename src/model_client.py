from dataclasses import dataclass
from src.data_handler import APIMessage
import litellm
from src.utils.logging import get_logger
from typing import Any
import time

logger = get_logger(__name__)


class MalformedResponseError(Exception):
    """Exception raised when the response from the model is malformed."""

    def __init__(self, message: str) -> None:
        """Initialize the exception.

        Args:
            message (str): The message to include in the exception.

        """
        super().__init__(message)


@dataclass
class ModelConfig:
    """Configuration for the model client.

    Attributes:
        model_name (str): The name of the model to use.
        temperature (float): The temperature to use for the model.
        max_tokens (int): The maximum number of tokens to generate.
        api_key (str | None): The API key to use for the model. Optional.
        api_base (str | None): The base URL for the API. Optional.
        max_retries (int): The maximum number of retries to make.
        timeout (int): The timeout for each request.

    """

    # 1. Identity
    model_name: str  # e.g., "openai/gpt-4o", "gemini/gemini-1.5-pro"

    temperature: float = 0.0
    max_tokens: int = 4096

    # 3. Network & Routing
    api_key: str | None = None
    api_base: str | None = None

    # 4. Resilience
    max_retries: int = 3
    timeout: int = 120
    pacing_delay: float = 0.0
    initial_backoff: float = 2.0
    backoff_factor: float = 2.0


class ModelClient:
    """Client for interacting with a SOTA model.

    Attributes:
        config (ModelConfig): The configuration for the model client.

    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the model client.

        Args:
            config (ModelConfig): The configuration for the model client.

        """
        self.config = config

    def generate_response(self, messages: list[APIMessage]) -> str:
        """Generate a response from the model.

        Args:
            messages (list[APIMessage]): The messages to send to the model.

        Returns:
            str: The response from the model.

        """
        attempt = 0
        current_backoff = self.config.initial_backoff

        if self.config.pacing_delay > 0:
            time.sleep(self.config.pacing_delay)

        while attempt <= self.config.max_retries:
            try:
                response = litellm.completion(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    api_base=self.config.api_base,
                    api_key=self.config.api_key,
                    num_retries=0,
                    timeout=self.config.timeout,
                )
                return self._unpack_response(response)

            except litellm.exceptions.RateLimitError as e:
                attempt += 1
                if attempt > self.config.max_retries:
                    logger.error(
                        f"Rate limit error exhausted after {attempt - 1} retries "
                        f"for model {self.config.model_name}: {e}",
                    )
                    return ""

                logger.warning(
                    f"Rate limit hit for {self.config.model_name}. "
                    f"Backing off for {current_backoff}s "
                    f"(Attempt {attempt}/{self.config.max_retries})",
                )
                time.sleep(current_backoff)
                current_backoff *= self.config.backoff_factor

            except litellm.exceptions.Timeout as e:
                logger.error(f"Timeout error for model {self.config.model_name}: {e}")
                return ""
            except MalformedResponseError as e:
                logger.error(
                    f"Malformed response error for model {self.config.model_name}: {e}",
                )
                return ""
            except Exception as e:
                logger.error(
                    f"Unexpected API error for model {self.config.model_name}: {e}",
                )
                return ""
        return ""

    @staticmethod
    def _unpack_response(response: Any) -> str:
        """Extract the text content from the model's response.

        Args:
            response (Any): The response from litellm.

        Raises:
            MalformedResponseError: If the response is not valid.

        Returns:
            str: The response content.

        """
        if not response:
            raise MalformedResponseError("Response is None or empty.")

        try:
            if hasattr(response, "choices"):
                content = response.choices[0].message.content
            else:
                content = response["choices"][0]["message"]["content"]

            return str(content)

        except (KeyError, TypeError, IndexError, AttributeError) as e:
            raise MalformedResponseError(f"Response is malformed: {e}") from e
