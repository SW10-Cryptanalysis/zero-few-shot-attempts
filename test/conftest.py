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


@pytest.fixture
def mock_components(mocker):
    """Provides mocked versions of the three main dependencies."""
    from src.data_handler import CipherSample, CipherMetadata
    # Data Handler Mock
    handler = mocker.Mock()
    # Mock sample generation
    s1 = CipherSample(
        "id_1", "SECRET", "XYXWYZ", {"A": ["B"]}, CipherMetadata(6, False, "test")
    )
    s2 = CipherSample(
        "id_2", "SECRET", "XYXWYZ", {"A": ["B"]}, CipherMetadata(6, False, "test")
    )

    handler.get_batch.return_value = [[s1, s2]]
    handler.dataset = [s1, s2]
    handler.format_prompt.return_value = [{"role": "user", "content": "prompt"}]

    # Client Mock
    client = mocker.Mock()
    client.config.model_name = "test-provider/test-model"
    client.generate_response.return_value = "PREDICTION"

    # Evaluator Mock
    evaluator = mocker.Mock()
    eval_mock = mocker.Mock()
    eval_mock.ser = 0.1
    eval_mock.is_exact_match = False
    eval_mock.raw_output = "PREDICTION"
    eval_mock.cleaned_prediction = "PREDICTION"
    evaluator.evaluate.return_value = eval_mock

    return handler, client, evaluator


@pytest.fixture
def pipeline(mock_components, tmp_path):
    """Provides a pipeline instance with mocked dependencies."""
    from src.experiment_pipeline import ExperimentPipeline
    handler, client, evaluator = mock_components
    return ExperimentPipeline(
        handler=handler,
        client=client,
        evaluator=evaluator,
        output_dir=tmp_path / "results",
    )
