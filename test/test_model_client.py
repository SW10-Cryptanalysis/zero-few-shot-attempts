import pytest
from dataclasses import dataclass
from typing import Any
import litellm
import openai
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


# =====================================================================
# 1. Routing & Execution Kwargs
# =====================================================================


@dataclass
class RoutingTestCase:
    name: str
    backend: str


@pytest.mark.parametrize(
    "tc",
    [
        RoutingTestCase("LiteLLM Backend Execution", "litellm"),
        RoutingTestCase("OpenAI SDK Backend Execution", "openai"),
        RoutingTestCase("OpenRouter Alias Execution", "openrouter"),
    ],
    ids=lambda tc: tc.name,
)
def test_backend_routing_and_kwargs(mocker, tc: RoutingTestCase):
    """Guarantees full coverage of routing logic and kwarg expansion."""
    config = ModelConfig(
        model_name="test-model",
        backend=tc.backend,
        api_key="sk-dummy-key",  # <-- Added dummy key
        extra_kwargs={"seed": 42},
    )
    client = ModelClient(config=config)

    mock_response = {"choices": [{"message": {"content": "Success"}}]}

    if tc.backend == "litellm":
        mock_call = mocker.patch(
            "src.model_client.litellm.completion", return_value=mock_response
        )
    else:
        mock_call = mocker.patch.object(
            client.openai_client.chat.completions, "create", return_value=mock_response
        )

    result = client.generate_response([{"role": "user", "content": "test"}])

    assert result == "Success"
    mock_call.assert_called_once()

    kwargs = mock_call.call_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["seed"] == 42


# =====================================================================
# 2. General Exception Handling
# =====================================================================


@dataclass
class ExceptionHandlingTestCase:
    name: str
    backend: str
    mock_effect: Any
    expected_output: str | None
    log_message_contains: str


@pytest.mark.parametrize(
    "tc",
    [
        ExceptionHandlingTestCase(
            name="LiteLLM Timeout",
            backend="litellm",
            mock_effect=litellm.exceptions.Timeout(
                "Timeout", model="", llm_provider=""
            ),
            expected_output=None,
            log_message_contains="Unexpected API error",
        ),
        ExceptionHandlingTestCase(
            name="Malformed Response Structure",
            backend="openai",
            mock_effect={"invalid": "structure"},
            expected_output=None,
            log_message_contains="Malformed response error",
        ),
        ExceptionHandlingTestCase(
            name="Generic System Exception",
            backend="litellm",
            mock_effect=ValueError("Unexpected system failure"),
            expected_output=None,
            log_message_contains="Unexpected API error",
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_generate_response_exceptions(mocker, tc: ExceptionHandlingTestCase):
    config = ModelConfig(
        model_name="test-model",
        max_retries=0,
        backend=tc.backend,
        api_key="sk-dummy-key",  # <-- Added dummy key
    )
    client = ModelClient(config=config)

    mock_kwargs = (
        {"side_effect": tc.mock_effect}
        if isinstance(tc.mock_effect, Exception)
        else {"return_value": tc.mock_effect}
    )

    # 2. Apply the mock to the correct target using unpacking
    if tc.backend == "litellm":
        mocker.patch("src.model_client.litellm.completion", **mock_kwargs)
    else:
        mocker.patch.object(
            client.openai_client.chat.completions, "create", **mock_kwargs
        )

    mock_logger = mocker.patch("src.model_client.logger.error")
    result = client.generate_response([{"role": "user", "content": "test"}])

    assert result == tc.expected_output
    assert mock_logger.call_count == 1
    assert tc.log_message_contains in mock_logger.call_args[0][0]


# =====================================================================
# 3. Pacing & Exponential Backoff Loop
# =====================================================================


@dataclass
class PacingBackoffTestCase:
    name: str
    backend: str
    api_responses: list[Any]
    pacing_delay: float
    max_retries: int
    expected_output: str | None
    expected_sleep_calls: list[float]
    expected_api_calls: int


@pytest.mark.parametrize(
    "tc",
    [
        PacingBackoffTestCase(
            name="LiteLLM: Success on first try with pacing",
            backend="litellm",
            api_responses=[{"choices": [{"message": {"content": "Success"}}]}],
            pacing_delay=0.5,
            max_retries=2,
            expected_output="Success",
            expected_sleep_calls=[0.5],
            expected_api_calls=1,
        ),
        PacingBackoffTestCase(
            name="LiteLLM: Recover after RateLimitError",
            backend="litellm",
            api_responses=[
                litellm.exceptions.RateLimitError(
                    "Rate limit", model="", llm_provider=""
                ),
                {"choices": [{"message": {"content": "Recovered"}}]},
            ],
            pacing_delay=0.0,
            max_retries=2,
            expected_output="Recovered",
            expected_sleep_calls=[0.0, 20.0],
            expected_api_calls=2,
        ),
        PacingBackoffTestCase(
            name="OpenAI: Recover after RateLimitError",
            backend="openai",
            api_responses=[
                "OPENAI_RATE_LIMIT",
                {"choices": [{"message": {"content": "Recovered"}}]},
            ],
            pacing_delay=0.0,
            max_retries=2,
            expected_output="Recovered",
            expected_sleep_calls=[0.0, 20.0],
            expected_api_calls=2,
        ),
        PacingBackoffTestCase(
            name="LiteLLM: Exhaust all retries with RateLimitError math",
            backend="litellm",
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
            expected_output=None,
            expected_sleep_calls=[0.0, 20.0, 40.0, 80.0],
            expected_api_calls=3,
        ),
        PacingBackoffTestCase(
            name="OpenAI: Recover after APIConnectionError",
            backend="openai",
            api_responses=[
                "OPENAI_NETWORK_ERROR",
                {"choices": [{"message": {"content": "Network Restored"}}]},
            ],
            pacing_delay=0.0,
            max_retries=2,
            expected_output="Network Restored",
            expected_sleep_calls=[0.0, 2.0],
            expected_api_calls=2,
        ),
        PacingBackoffTestCase(
            name="LiteLLM: Exhaust all retries with Network Error backoff math",
            backend="litellm",
            api_responses=[
                litellm.exceptions.APIConnectionError(
                    "Dropped", model="", llm_provider=""
                ),
                litellm.exceptions.APIConnectionError(
                    "Dropped", model="", llm_provider=""
                ),
                litellm.exceptions.APIConnectionError(
                    "Dropped", model="", llm_provider=""
                ),
            ],
            pacing_delay=0.0,
            max_retries=2,
            expected_output=None,
            expected_sleep_calls=[0.0, 2.0, 4.0, 8.0],
            expected_api_calls=3,
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_backoff_and_pacing_logic(mocker, tc: PacingBackoffTestCase):
    resolved_responses = []
    for resp in tc.api_responses:
        if resp == "OPENAI_RATE_LIMIT":
            resolved_responses.append(
                openai.RateLimitError("Limit", response=mocker.Mock(), body=None)
            )
        elif resp == "OPENAI_NETWORK_ERROR":
            resolved_responses.append(openai.APIConnectionError(request=mocker.Mock()))
        else:
            resolved_responses.append(resp)

    config = ModelConfig(
        model_name="test-model",
        max_retries=tc.max_retries,
        pacing_delay=tc.pacing_delay,
        initial_backoff=2.0,
        backoff_factor=2.0,
        backend=tc.backend,
        api_key="sk-dummy-key",  # <-- Added dummy key
    )
    client = ModelClient(config=config)
    mock_sleep = mocker.patch("src.model_client.time.sleep")

    if tc.backend == "litellm":
        mock_completion = mocker.patch(
            "src.model_client.litellm.completion", side_effect=resolved_responses
        )
    else:
        mock_completion = mocker.patch.object(
            client.openai_client.chat.completions,
            "create",
            side_effect=resolved_responses,
        )

    result = client.generate_response([{"role": "user", "content": "test"}])

    assert result == tc.expected_output
    assert mock_completion.call_count == tc.expected_api_calls

    actual_sleeps = [call.args[0] for call in mock_sleep.call_args_list]
    assert actual_sleeps == tc.expected_sleep_calls


# =====================================================================
# 4. Response Unpacking Edge Cases
# =====================================================================


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
            name="None response",
            response_input=None,
            expected_output=None,
            expect_exception=True,
        ),
        UnpackTestCase(
            name="Malformed Dictionary (Missing message block)",
            response_input={"choices": [{"wrong_key": "value"}]},
            expected_output=None,
            expect_exception=True,
        ),
        UnpackTestCase(
            name="TypeError from invalid choices type",
            response_input={"choices": "not_a_list"},
            expected_output=None,
            expect_exception=True,
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_unpack_response(tc: UnpackTestCase):
    if tc.expect_exception:
        with pytest.raises(MalformedResponseError):
            ModelClient._unpack_response(tc.response_input)
    else:
        result = ModelClient._unpack_response(tc.response_input)
        assert result == tc.expected_output


def test_generate_response_while_loop_exhaustion(mocker):
    """Guarantees the while-loop cleanly exits if max_retries is negative."""
    config = ModelConfig(
        model_name="test-model",
        max_retries=-1,
        api_key="sk-dummy-key",  # <-- Added dummy key
    )
    client = ModelClient(config)
    result = client.generate_response([{"role": "user", "content": "test"}])
    assert result is None
