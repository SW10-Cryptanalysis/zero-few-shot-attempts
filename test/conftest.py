import sys
from unittest.mock import MagicMock
import pytest


# Define REAL exception classes that inherit from BaseException
class MockTimeoutError(Exception):
    def __init__(self, message, model, llm_provider):
        super().__init__(message, model, llm_provider)


class MockRateLimitError(Exception):
    def __init__(self, message, model, llm_provider):
        super().__init__(message, model, llm_provider)


# Setup the global mock
MOCK_LITELLM = MagicMock()
MOCK_EXCEPTIONS = MagicMock()

# Assign the real classes to the mock attributes
MOCK_EXCEPTIONS.Timeout = MockTimeoutError
MOCK_EXCEPTIONS.RateLimitError = MockRateLimitError
MOCK_LITELLM.exceptions = MOCK_EXCEPTIONS


def pytest_configure(config):
    sys.modules["litellm"] = MOCK_LITELLM
    sys.modules["litellm.exceptions"] = MOCK_EXCEPTIONS


@pytest.fixture(autouse=True)
def reset_litellm_mock():
    MOCK_LITELLM.completion.reset_mock()
    MOCK_LITELLM.completion.side_effect = None
    MOCK_LITELLM.completion.return_value = MagicMock()
    yield
