import os
os.environ["LITELLM_DISABLE_UPDATE_CHECK"] = "True"
os.environ["LITELLM_TELEMETRY"] = "False"
import pytest
from dataclasses import dataclass
from typing import Any
import litellm
from src.model_client import ModelClient, ModelConfig
from src.data_handler import APIMessage

# =====================================================================
# 1. Dummy Classes to replace MagicMock for litellm responses
# =====================================================================


@dataclass
class DummyMessage:
    content: str | None


@dataclass
class DummyChoice:
    message: DummyMessage


@dataclass
class DummyResponse:
    choices: list[DummyChoice]


# =====================================================================
# 2. Test Dataclasses
# =====================================================================


@dataclass
class UnpackTestCase:
    name: str
    response_input: Any
    expected_output: str | None
    expect_exception: bool


@dataclass
class GenerateTestCase:
    name: str
    mock_effect: Any  # Can be a return value or an Exception
    expected_output: str
    expected_log_type: type[Exception] | None


# =====================================================================
# 3. Fixtures
# =====================================================================


@pytest.fixture
def base_config() -> ModelConfig:
    """Provides a standard config for testing."""
    return ModelConfig(
        model_name="test-model",
        temperature=0.0,
        max_tokens=100,
        max_retries=1,
        timeout=10,
    )


@pytest.fixture
def client(base_config: ModelConfig) -> ModelClient:
    """Provides an instance of the ModelClient."""
    return ModelClient(config=base_config)


# =====================================================================
# 4. Unit Tests for Static Method _unpack_response
# =====================================================================


@pytest.mark.parametrize(
    "tc",
    [
        UnpackTestCase(
            name="Valid Object-based Response (LiteLLM standard)",
            response_input=DummyResponse([DummyChoice(DummyMessage("Decrypted!"))]),
            expected_output="Decrypted!",
            expect_exception=False,
        ),
        UnpackTestCase(
            name="Valid Dictionary-based Response (Raw JSON fallback)",
            response_input={"choices": [{"message": {"content": "Dict Decrypted!"}}]},
            expected_output="Dict Decrypted!",
            expect_exception=False,
        ),
        UnpackTestCase(
            name="None Response",
            response_input=None,
            expected_output=None,
            expect_exception=True,
        ),
        UnpackTestCase(
            name="Malformed Dictionary (Missing 'message')",
            response_input={"choices": [{"wrong_key": "value"}]},
            expected_output=None,
            expect_exception=True,  # Catches KeyError
        ),
        UnpackTestCase(
            name="Empty Choices List",
            response_input={"choices": []},
            expected_output=None,
            expect_exception=True,  # Catches IndexError
        ),
        UnpackTestCase(
            name="Content is explicitly None",
            response_input=DummyResponse([DummyChoice(DummyMessage(None))]),
            expected_output="None",  # str(None) -> "None" based on your current str() casting
            expect_exception=False,
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_unpack_response(tc: UnpackTestCase):
    """Test the static unpacking method handles objects, dicts, and malformed data."""
    if tc.expect_exception:
        # Note: adjust the import path for MalformedResponseError if needed
        with pytest.raises(Exception) as exc_info:
            ModelClient._unpack_response(tc.response_input)
        assert "MalformedResponseError" in exc_info.typename
    else:
        result = ModelClient._unpack_response(tc.response_input)
        assert result == tc.expected_output


# =====================================================================
# 5. Unit Tests for generate_response
# =====================================================================

# Initialize mock exceptions required by litellm
timeout_err = litellm.exceptions.Timeout(
    "Timeout hit", model="test", llm_provider="test"
)
rate_limit_err = litellm.exceptions.RateLimitError(
    "Rate limit hit", model="test", llm_provider="test"
)


@pytest.mark.parametrize(
    "tc",
    [
        GenerateTestCase(
            name="Successful Generation",
            mock_effect=DummyResponse([DummyChoice(DummyMessage("Success Output"))]),
            expected_output="Success Output",
            expected_log_type=None,
        ),
        GenerateTestCase(
            name="Timeout Exception",
            mock_effect=timeout_err,
            expected_output="",
            expected_log_type=litellm.exceptions.Timeout,
        ),
        GenerateTestCase(
            name="Rate Limit Exception",
            mock_effect=rate_limit_err,
            expected_output="",
            expected_log_type=litellm.exceptions.RateLimitError,
        ),
        GenerateTestCase(
            name="Generic Exception",
            mock_effect=ValueError("Some random internal error"),
            expected_output="",
            expected_log_type=Exception,  # Generic catch-all
        ),
        GenerateTestCase(
            name="Malformed Response Exception",
            mock_effect={
                "bad": "data"
            },  # Will cause _unpack_response to raise MalformedResponseError
            expected_output="",
            expected_log_type=Exception,  # Caught by the MalformedResponseError block
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_generate_response(client: ModelClient, mocker, tc: GenerateTestCase):
    """Test that generate_response correctly maps LiteLLM responses and exceptions."""

    # 1. Mock the litellm.completion call
    if isinstance(tc.mock_effect, Exception):
        mocker.patch("litellm.completion", side_effect=tc.mock_effect)
    else:
        mocker.patch("litellm.completion", return_value=tc.mock_effect)

    # 2. Spy on the logger to ensure errors are being logged appropriately
    mock_logger = mocker.patch("src.model_client.logger.error")

    # 3. Execute the method
    dummy_messages: list[APIMessage] = [{"role": "user", "content": "test"}]
    result = client.generate_response(dummy_messages)

    # 4. Assertions
    assert result == tc.expected_output

    if tc.expected_log_type:
        assert mock_logger.call_count == 1
        # Check that the log message contains the model name
        assert client.config.model_name in mock_logger.call_args[0][0]
    else:
        assert mock_logger.call_count == 0
