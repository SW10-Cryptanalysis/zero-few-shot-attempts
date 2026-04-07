from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

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
        api_key (str | None): The API key to use for the model. Optional.
        api_base (str | None): The base URL for the API. Optional.
        max_retries (int): The maximum number of retries to make.
        timeout (int): The timeout for each request.
        pacing_delay (float): Seconds between API calls.
        initial_backoff (float): Initial backoff time if API call fails.
        backoff_factor (float): Backoff time multiplier.

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


class BaseModelClient(ABC):
    """Abstract base class for interacting with a SOTA model.

    Attributes:
        config (ModelConfig): The configuration for the model client.

    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the model client base.

        Args:
            config (ModelConfig): The configuration for the model client.

        """
        self.config = config

    @abstractmethod
    def generate_response(self, messages: list[APIMessage]) -> str | None:
        """Generate a response from the model.

        Args:
            messages (list[APIMessage]): The messages to send to the model.

        Returns:
            str | None: The response from the model, or None on fatal failure.

        """
        pass

    @staticmethod
    def _unpack_response(response: Any) -> str:
        """Extract the text content from the model's response.

        Args:
            response (Any): The response from the API.

        Raises:
            MalformedResponseError: If the response is not valid.

        Returns:
            str: The response content.

        """
        if not response or response == "None":
            raise MalformedResponseError("Response is None or empty.")

        try:
            if hasattr(response, "choices"):
                content = response.choices[0].message.content
            else:
                content = response["choices"][0]["message"]["content"]

            return str(content)

        except (KeyError, TypeError, IndexError, AttributeError) as e:
            raise MalformedResponseError(f"Response is malformed: {e}") from e




