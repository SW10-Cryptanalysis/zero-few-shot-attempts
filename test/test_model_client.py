import pytest
from dataclasses import dataclass
from typing import Any
import litellm
from src.model_client import ModelClient, ModelConfig, MalformedResponseError


@dataclass
class DummyMessage:
    content: str | None


@dataclass
class DummyChoice:
    message: DummyMessage


@dataclass
class DummyResponse:
    choices: list[DummyChoice]


@dataclass
class ExceptionHandlingTestCase:
    name: str
    mock_effect: Any
    expected_output: str
    log_message_contains: str


@pytest.fixture
def client() -> ModelClient:
    config = ModelConfig(model_name="test-model", pacing_delay=0.0, max_retries=0)
    return ModelClient(config=config)


@pytest.mark.parametrize(
    "tc",
    [
        ExceptionHandlingTestCase(
            name="Timeout Exception",
            mock_effect=litellm.exceptions.Timeout(
                "Timeout", model="", llm_provider=""
            ),
            expected_output="",
            log_message_contains="Timeout error",
        ),
        ExceptionHandlingTestCase(
            name="Malformed Response Exception",
            mock_effect={"invalid": "structure"},
            expected_output="",
            log_message_contains="Malformed response error",
        ),
        ExceptionHandlingTestCase(
            name="Generic Exception",
            mock_effect=ValueError("Unexpected system failure"),
            expected_output="",
            log_message_contains="Unexpected API error",
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_generate_response_exceptions(
    client: ModelClient, mocker, tc: ExceptionHandlingTestCase
):
    patch_target = "src.model_client.litellm.completion"

    if isinstance(tc.mock_effect, Exception):
        mocker.patch(patch_target, side_effect=tc.mock_effect)
    else:
        mocker.patch(patch_target, return_value=tc.mock_effect)

    mock_logger = mocker.patch("src.model_client.logger.error")

    result = client.generate_response([{"role": "user", "content": "test"}])

    assert result == tc.expected_output
    assert mock_logger.call_count == 1
    assert tc.log_message_contains in mock_logger.call_args[0][0]


@dataclass
class UnpackEdgeCaseTestCase:
    name: str
    response_input: Any
    expected_exception_msg: str


@pytest.mark.parametrize(
    "tc",
    [
        UnpackEdgeCaseTestCase(
            name="None response",
            response_input=None,
            expected_exception_msg="Response is None or empty.",
        ),
        UnpackEdgeCaseTestCase(
            name="Empty dictionary response",
            response_input={},
            expected_exception_msg="Response is None or empty.",
        ),
        UnpackEdgeCaseTestCase(
            name="KeyError from missing message block",
            response_input={"choices": [{"wrong_key": "value"}]},
            expected_exception_msg="Response is malformed",
        ),
        UnpackEdgeCaseTestCase(
            name="IndexError from empty choices list",
            response_input={"choices": []},
            expected_exception_msg="Response is malformed",
        ),
        UnpackEdgeCaseTestCase(
            name="TypeError from invalid choices type",
            response_input={"choices": "not_a_list"},
            expected_exception_msg="Response is malformed",
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_unpack_response_edge_cases(tc: UnpackEdgeCaseTestCase):
    with pytest.raises(MalformedResponseError) as exc_info:
        ModelClient._unpack_response(tc.response_input)
    assert tc.expected_exception_msg in str(exc_info.value)


@dataclass
class PacingBackoffTestCase:
    name: str
    api_responses: list[Any]
    pacing_delay: float
    max_retries: int
    expected_output: str
    expected_sleep_calls: list[float]
    expected_api_calls: int


@pytest.fixture
def backoff_config() -> ModelConfig:
    return ModelConfig(
        model_name="test-model",
        max_retries=2,
        pacing_delay=0.5,
        initial_backoff=2.0,
        backoff_factor=2.0,
    )


@pytest.mark.parametrize(
    "tc",
    [
        PacingBackoffTestCase(
            name="Success on first try with pacing",
            api_responses=[{"choices": [{"message": {"content": "Success"}}]}],
            pacing_delay=0.5,
            max_retries=2,
            expected_output="Success",
            expected_sleep_calls=[0.5],
            expected_api_calls=1,
        ),
        PacingBackoffTestCase(
            name="Recover after one rate limit error",
            api_responses=[
                litellm.exceptions.RateLimitError(
                    "Rate limit", model="", llm_provider=""
                ),
                {"choices": [{"message": {"content": "Recovered"}}]},
            ],
            pacing_delay=0.0,
            max_retries=2,
            expected_output="Recovered",
            expected_sleep_calls=[2.0],
            expected_api_calls=2,
        ),
        PacingBackoffTestCase(
            name="Exhaust all retries with exponential backoff",
            api_responses=[
                litellm.exceptions.RateLimitError(
                    "Rate limit", model="", llm_provider=""
                ),
                litellm.exceptions.RateLimitError(
                    "Rate limit", model="", llm_provider=""
                ),
                litellm.exceptions.RateLimitError(
                    "Rate limit", model="", llm_provider=""
                ),
            ],
            pacing_delay=0.0,
            max_retries=2,
            expected_output="",
            expected_sleep_calls=[2.0, 4.0],
            expected_api_calls=3,
        ),
        PacingBackoffTestCase(
            name="Pacing + Multiple Rate Limits combined",
            api_responses=[
                litellm.exceptions.RateLimitError(
                    "Rate limit", model="", llm_provider=""
                ),
                litellm.exceptions.RateLimitError(
                    "Rate limit", model="", llm_provider=""
                ),
                {"choices": [{"message": {"content": "Finally"}}]},
            ],
            pacing_delay=1.0,
            max_retries=3,
            expected_output="Finally",
            expected_sleep_calls=[1.0, 2.0, 4.0],
            expected_api_calls=3,
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_backoff_and_pacing_logic(
    client: ModelClient, mocker, tc: PacingBackoffTestCase
):
    client.config.pacing_delay = tc.pacing_delay
    client.config.max_retries = tc.max_retries

    mock_sleep = mocker.patch("src.model_client.time.sleep")

    patch_target = "src.model_client.litellm.completion"
    mock_completion = mocker.patch(patch_target, side_effect=tc.api_responses)

    result = client.generate_response([{"role": "user", "content": "test"}])

    assert result == tc.expected_output
    assert mock_completion.call_count == tc.expected_api_calls

    actual_sleeps = [call.args[0] for call in mock_sleep.call_args_list]
    assert actual_sleeps == tc.expected_sleep_calls


@dataclass
class UnpackTestCase:
    name: str
    response_input: Any
    expected_output: str | None
    expect_exception: bool


@pytest.mark.parametrize(
    "tc",
    [
        UnpackTestCase(
            name="Valid Dictionary-based Response",
            response_input={"choices": [{"message": {"content": "Decrypted!"}}]},
            expected_output="Decrypted!",
            expect_exception=False,
        ),
        UnpackTestCase(
            name="Valid Object-based Response",
            response_input=DummyResponse(
                [DummyChoice(DummyMessage("Object Decrypted!"))]
            ),
            expected_output="Object Decrypted!",
            expect_exception=False,
        ),
        UnpackTestCase(
            name="Malformed Dictionary (Missing message)",
            response_input={"choices": [{"wrong_key": "value"}]},
            expected_output=None,
            expect_exception=True,
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_unpack_response(tc: UnpackTestCase):
    if tc.expect_exception:
        with pytest.raises(Exception) as exc_info:
            ModelClient._unpack_response(tc.response_input)
        assert "MalformedResponseError" in exc_info.typename
    else:
        result = ModelClient._unpack_response(tc.response_input)
        assert result == tc.expected_output


def test_generate_response_while_loop_exhaustion(client: ModelClient, mocker):
    client.config.max_retries = -1
    result = client.generate_response([{"role": "user", "content": "test"}])
    assert result == ""
