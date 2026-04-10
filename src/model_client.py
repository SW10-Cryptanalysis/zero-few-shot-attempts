import time
from dataclasses import dataclass, field
from typing import Any

import litellm
import openai
from openai import OpenAI

from src.data_handler import APIMessage
from src.utils.logging import get_logger

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
        api_key (str | None): The API key to use for the model.
        api_base (str | None): The base URL for the API.
        max_retries (int): The maximum number of retries to make.
        timeout (int): The timeout for each request.
        pacing_delay (float): Seconds between API calls.
        initial_backoff (float): Initial backoff time if API call fails.
        backoff_factor (float): Backoff time multiplier.
        backend (str): The execution engine to use ('openai' or 'litellm').
        extra_kwargs (dict[str, Any]): Additional parameters to pass to the API.

    """

    model_name: str
    temperature: float = 0.0
    max_tokens: int = 4096
    api_key: str | None = None
    api_base: str | None = None
    max_retries: int = 3
    timeout: int = 120
    pacing_delay: float = 0.0
    initial_backoff: float = 2.0
    backoff_factor: float = 2.0
    backend: str = "openai"
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


class ModelClient:
    """A unified client for interacting with SOTA models via OpenAI or LiteLLM."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the unified model client.

        Args:
            config (ModelConfig): The configuration parameters.

        """
        self.config = config

        if self.config.backend == "openai" or self.config.backend == "openrouter":
            self.openai_client = OpenAI(
                base_url=self.config.api_base,
                api_key=self.config.api_key,
            )

    def _execute_openai(self, messages: list[APIMessage]) -> Any:
        """Execute the request using the official OpenAI SDK."""
        return self.openai_client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,  # type: ignore
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            **self.config.extra_kwargs,
        )

    def _execute_litellm(self, messages: list[APIMessage]) -> Any:
        """Execute the request using the LiteLLM router."""
        return litellm.completion(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=self.config.api_key,
            api_base=self.config.api_base,
            num_retries=0,
            timeout=self.config.timeout,
            **self.config.extra_kwargs,
        )

    def _execute(self, messages: list[APIMessage]) -> Any:
        """Execute the request using the configured backend."""
        if self.config.backend == "openai" or self.config.backend == "openrouter":
            return self._execute_openai(messages)
        return self._execute_litellm(messages)

    def generate_response(self, messages: list[APIMessage]) -> str | None:
        """Generate a response using the configured backend with unified backoff.

        Args:
            messages (list[APIMessage]): The messages to send.

        Returns:
            str | None: The response text, or None on fatal error.

        """
        attempt = 0
        current_backoff = self.config.initial_backoff

        time.sleep(self.config.pacing_delay)

        rate_limit_errors = (openai.RateLimitError, litellm.exceptions.RateLimitError)
        network_errors = (
            openai.APIConnectionError,
            openai.InternalServerError,
            litellm.exceptions.APIConnectionError,
            litellm.exceptions.ServiceUnavailableError,
        )

        while attempt <= self.config.max_retries:
            try:
                response = self._execute(messages)

                return self._unpack_response(response)

            except rate_limit_errors as e:
                logger.warning(f"Rate Limit Hit! Details: {e}")
                # Attempt to retrieve the quota reset time from the error message
                pause = 20 * max(attempt * self.config.backoff_factor, 1)
                logger.info(f"Sleeping for {pause} seconds to let quotas reset...")
                time.sleep(pause)
                attempt += 1

            except network_errors as e:
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

    @staticmethod
    def _unpack_response(response: Any) -> str:
        """Extract the text content from the API response object.

        Args:
            response (Any): The response object.

        Raises:
            MalformedResponseError: If the response is not valid.

        Returns:
            str: The response content.

        """
        if not response:
            raise MalformedResponseError("Response is None or empty.")

        try:
            if hasattr(response, "choices"):
                content = str(response.choices[0].message.content)
            else:
                content = str(response["choices"][0]["message"]["content"])

            return str(content).replace("\n", " ").replace(",", "")

        except (KeyError, TypeError, IndexError, AttributeError) as e:
            raise MalformedResponseError(f"Response is malformed: {e}") from e
