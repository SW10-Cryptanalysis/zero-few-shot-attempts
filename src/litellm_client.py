from src.model_client import BaseModelClient, MalformedResponseError, APIMessage
import time
import litellm

from src.utils.logging import get_logger

logger = get_logger(__name__)

class LiteLLMClient(BaseModelClient):
    """Client for interacting with models using the LiteLLM router."""

    def generate_response(self, messages: list[APIMessage]) -> str | None:
        """Generate a response using LiteLLM with resilience and backoff."""
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
                    api_key=self.config.api_key,
                    api_base=self.config.api_base,
                    num_retries=0,
                    timeout=self.config.timeout,
                )
                return self._unpack_response(response)

            except litellm.exceptions.RateLimitError as e:
                logger.warning(
                    f"Rate Limit Hit! The API is demanding a pause. Details: {e}",
                )

                long_pause = 10 * max(attempt * self.config.backoff_factor, 1)
                logger.info(f"Sleeping for {long_pause} seconds to let quotas reset...")
                time.sleep(long_pause)
                attempt += 1

            except (
                litellm.exceptions.APIConnectionError,
                litellm.exceptions.ServiceUnavailableError,
            ) as e:
                logger.warning(
                    f"Network error (Attempt {attempt + 1}/"
                    f"{self.config.max_retries + 1}): {e}",
                )
                time.sleep(current_backoff)
                current_backoff *= self.config.backoff_factor
                attempt += 1

            except litellm.exceptions.Timeout as e:
                logger.error(f"Timeout error for model {self.config.model_name}: {e}")
                return None
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
