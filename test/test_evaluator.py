import pytest
from dataclasses import dataclass
from src.evaluator import Evaluator

# =====================================================================
# 1. Test Dataclasses
# =====================================================================


@dataclass
class CleanPredictionTestCase:
    name: str
    raw_output: str
    expected_clean: str


@dataclass
class SERTestCase:
    name: str
    prediction: str
    ground_truth: str
    expected_ser: float


@dataclass
class EvaluateTestCase:
    name: str
    raw_output: str
    ground_truth: str
    expected_ser: float
    expected_exact_match: bool


# =====================================================================
# 2. Fixtures
# =====================================================================


@pytest.fixture
def evaluator() -> Evaluator:  # Replace Evaluator with your actual import
    """Provides a fresh instance of the Evaluator for each test."""
    return Evaluator()


# =====================================================================
# 3. Unit Tests
# =====================================================================


@pytest.mark.parametrize(
    "tc",
    [
        CleanPredictionTestCase(name="Empty string", raw_output="", expected_clean=""),
        CleanPredictionTestCase(
            name="Perfect output without filler",
            raw_output="HELLOWORLD" * 10,
            expected_clean="HELLOWORLD" * 10,
        ),
        CleanPredictionTestCase(
            name="Markdown code blocks",
            raw_output=f"```plaintext\n{'HELLOWORLD' * 10}\n```",
            expected_clean="HELLOWORLD" * 10,
        ),
        CleanPredictionTestCase(
            name="Conversational filler with double newlines",
            raw_output=f"Here is the text:\n\n{'ACTUALCIPHERTEXT' * 10}\n\nHope this helps!",
            expected_clean="ACTUALCIPHERTEXT" * 10,
        ),
        CleanPredictionTestCase(
            name="Markdown AND Conversational filler",
            raw_output=f"Sure thing! Here it is:\n\n```\n{'HELLOWORLD' * 10}\n```\n\nLet me know.",
            expected_clean="HELLOWORLD" * 10,
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_clean_prediction(evaluator: Evaluator, tc: CleanPredictionTestCase):
    """Test that the heuristic correctly isolates the plaintext."""
    result = evaluator._clean_prediction(tc.raw_output)
    assert result == tc.expected_clean


@pytest.mark.parametrize(
    "tc",
    [
        SERTestCase(
            name="Perfect Match",
            prediction="HELLO",
            ground_truth="HELLO",
            expected_ser=0.0,
        ),
        SERTestCase(
            name="Empty Prediction",
            prediction="",
            ground_truth="HELLO",
            expected_ser=1.0,
        ),
        SERTestCase(
            name="Empty Ground Truth",
            prediction="HELLO",
            ground_truth="",
            expected_ser=1.0,
        ),
        SERTestCase(
            name="One substitution error",
            prediction="HELLP",  # 'P' instead of 'O' (1 error)
            ground_truth="HELLO",
            expected_ser=0.2,  # 1 / 5 = 0.2
        ),
        SERTestCase(
            name="One deletion error",
            prediction="HELL",  # missing 'O' (1 error)
            ground_truth="HELLO",
            expected_ser=0.2,  # 1 / 5 = 0.2
        ),
        SERTestCase(
            name="Hallucination overflow (Cap at 1.0)",
            prediction="HELLO WORLD THIS IS WAY TOO LONG",
            ground_truth="HELLO",
            expected_ser=1.0,  # Distance is 27, 27/5 = 5.4 -> Capped to 1.0
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_calculate_ser(evaluator: Evaluator, tc: SERTestCase):
    """Test that the Symbol Error Rate math is correct and bounded."""
    # Using pytest.approx to avoid floating point precision issues (e.g., 0.200000000001)
    result = evaluator._calculate_ser(tc.prediction, tc.ground_truth)
    assert result == pytest.approx(tc.expected_ser, abs=1e-5)


@pytest.mark.parametrize(
    "tc",
    [
        EvaluateTestCase(
            name="End-to-End Perfect Match",
            raw_output="Decrypted:\n\nHELLOTHISISTHEPLAINTEXT\n\nDone.",
            ground_truth="HELLOTHISISTHEPLAINTEXT",
            expected_ser=0.0,
            expected_exact_match=True,
        ),
        EvaluateTestCase(
            name="End-to-End Partial Match",
            raw_output="```\nHELLP\n```",
            ground_truth="HELLO",
            expected_ser=0.2,
            expected_exact_match=False,
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_evaluate_integration(evaluator: Evaluator, tc: EvaluateTestCase):
    """Test the orchestration of cleaning and scoring."""
    result = evaluator.evaluate(tc.raw_output, tc.ground_truth)

    assert result.raw_output == tc.raw_output
    assert result.ground_truth == tc.ground_truth
    assert result.ser == pytest.approx(tc.expected_ser, abs=1e-5)

    assert result.is_exact_match == tc.expected_exact_match
