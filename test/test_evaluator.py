import pytest
from dataclasses import dataclass, field
from collections import Counter
from src.evaluator import Evaluator  # Replace with your actual import

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
class GetEmpiricalKeyTestCase:
    name: str
    mapping_freqs: dict[str, Counter]
    expected_empirical_key: dict[str, str]


@dataclass
class ExtractKeysTestCase:
    name: str
    prediction: str
    ground_truth: str
    ciphertext: str
    expected_true_key: dict[str, str] = field(default_factory=dict)
    expected_predicted_key: dict[str, str] = field(default_factory=dict)
    expect_error: bool = False


@dataclass
class SMERTestCase:
    name: str
    prediction: str
    ground_truth: str
    ciphertext: str
    expected_smer: float


@dataclass
class EvaluateTestCase:
    name: str
    raw_output: str
    ground_truth: str
    ciphertext: str
    expected_ser: float
    expected_smer: float
    expected_exact_match: bool


# =====================================================================
# 2. Fixtures
# =====================================================================


@pytest.fixture
def evaluator() -> Evaluator:
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
            raw_output="HELLOWORLD",
            expected_clean="HELLOWORLD",
        ),
        CleanPredictionTestCase(
            name="Markdown code blocks",
            raw_output="```plaintext\nHELLOWORLD\n```",
            expected_clean="HELLOWORLD",
        ),
        CleanPredictionTestCase(
            name="Conversational filler with double newlines",
            raw_output="Here is the text:\n\nACTUALCIPHERTEXT\n\nHope this helps!",
            expected_clean="ACTUALCIPHERTEXT",
            # Note: The updated _clean_prediction removes newlines but keeps structure
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_clean_prediction(evaluator: Evaluator, tc: CleanPredictionTestCase):
    """Test that the heuristic correctly isolates and formats the plaintext."""
    result = evaluator._clean_prediction(tc.raw_output)
    # If using your original longest-word heuristic, update expected_clean accordingly.
    # This assumes the new robust text cleaning method.
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
            name="One substitution",
            prediction="HELLP",
            ground_truth="HELLO",
            expected_ser=0.2,
        ),
        SERTestCase(
            name="One deletion",
            prediction="HELL",
            ground_truth="HELLO",
            expected_ser=0.2,
        ),
        SERTestCase(
            name="Hallucination overflow",
            prediction="HELLO WORLD TOO LONG",
            ground_truth="HELLO",
            expected_ser=1.0,
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_calculate_ser(evaluator: Evaluator, tc: SERTestCase):
    """Test that the Symbol Error Rate math is correct and bounded."""
    result = evaluator._calculate_ser(tc.prediction, tc.ground_truth)
    assert result == pytest.approx(tc.expected_ser, abs=1e-5)


@pytest.mark.parametrize(
    "tc",
    [
        GetEmpiricalKeyTestCase(
            name="Perfect 100% threshold",
            mapping_freqs={"1": Counter({"A": 100})},
            expected_empirical_key={"1": "A"},
        ),
        GetEmpiricalKeyTestCase(
            name="Exactly on 95% threshold",
            mapping_freqs={"1": Counter({"A": 95, "B": 5})},
            expected_empirical_key={"1": "A"},
        ),
        GetEmpiricalKeyTestCase(
            name="Fails 95% threshold",
            mapping_freqs={"1": Counter({"A": 94, "B": 6})},
            expected_empirical_key={},  # Discarded due to mapping instability
        ),
        GetEmpiricalKeyTestCase(
            name="Empty Counter (Truncation)",
            mapping_freqs={"1": Counter()},
            expected_empirical_key={},  # Handled safely without ZeroDivisionError
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_get_empirical_key(evaluator: Evaluator, tc: GetEmpiricalKeyTestCase):
    """Test the Strict Mapping Error Rate (SMER) threshold logic."""
    result = evaluator._get_emprirical_key(tc.mapping_freqs)
    assert result == tc.expected_empirical_key


@pytest.mark.parametrize(
    "tc",
    [
        ExtractKeysTestCase(
            name="Perfect full extraction",
            prediction="ABC",
            ground_truth="ABC",
            ciphertext="1 2 3",
            expected_true_key={"1": "A", "2": "B", "3": "C"},
            expected_predicted_key={"1": "A", "2": "B", "3": "C"},
        ),
        ExtractKeysTestCase(
            name="Premature Truncation",
            prediction="AB",
            ground_truth="ABC",
            ciphertext="1 2 3",
            expected_true_key={"1": "A", "2": "B", "3": "C"},
            expected_predicted_key={"1": "A", "2": "B"},  # Symbol 3 was not reached
        ),
        ExtractKeysTestCase(
            name="Many-to-One substitution handles correctly",
            prediction="ABA",
            ground_truth="ABA",
            ciphertext="1 2 3",
            expected_true_key={"1": "A", "2": "B", "3": "A"},
            expected_predicted_key={"1": "A", "2": "B", "3": "A"},
        ),
        ExtractKeysTestCase(
            name="ValueError on invalid ground truth",
            prediction="ABC",
            ground_truth="ABC",
            ciphertext="1 2 1",  # 1 maps to A and C
            expect_error=True,
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_extract_keys(evaluator: Evaluator, tc: ExtractKeysTestCase):
    """Test the iteration, truncation safety, and ground-truth validation."""
    if tc.expect_error:
        with pytest.raises(ValueError):
            evaluator._extract_keys(tc.prediction, tc.ground_truth, tc.ciphertext)
    else:
        true_key, pred_key = evaluator._extract_keys(
            tc.prediction, tc.ground_truth, tc.ciphertext
        )
        assert true_key == tc.expected_true_key
        assert pred_key == tc.expected_predicted_key


@pytest.mark.parametrize(
    "tc",
    [
        SMERTestCase(
            name="Empty Prediction",
            prediction="",
            ground_truth="HELLO",
            ciphertext="1 2 3 3 4",
            expected_smer=1.0,
        ),
        SMERTestCase(
            name="Perfect Match",
            prediction="HELLO",
            ground_truth="HELLO",
            ciphertext="1 2 3 3 4",
            expected_smer=0.0,  # 4 unique symbols, 0 errors
        ),
        SMERTestCase(
            name="One substitution error",
            prediction="HELLP",
            ground_truth="HELLO",
            ciphertext="1 2 3 3 4",
            expected_smer=0.25,  # Symbol 4 is wrong. 1 error / 4 unique symbols
        ),
        SMERTestCase(
            name="Inconsistent prediction (<95%)",
            prediction="AABA",
            ground_truth="AAAA",
            ciphertext="1 1 1 1",
            expected_smer=1.0,  # Symbol 1 maps to A(3) and B(1). 75% fails threshold. 1 error / 1 unique symbol
        ),
        SMERTestCase(
            name="Truncation penalty",
            prediction="HEL",
            ground_truth="HELLO",
            ciphertext="1 2 3 3 4",
            expected_smer=0.25,  # Symbol 4 never predicted. 1 error / 4 unique symbols
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_calculate_smer(evaluator: Evaluator, tc: SMERTestCase):
    """Test the overarching math of the Strict Mapping Error Rate."""
    result = evaluator._calculate_smer(tc.prediction, tc.ground_truth, tc.ciphertext)
    assert result == pytest.approx(tc.expected_smer, abs=1e-5)


@pytest.mark.parametrize(
    "tc",
    [
        EvaluateTestCase(
            name="End-to-End Perfect Match",
            raw_output="```\nHELLO\n```",
            ground_truth="HELLO",
            ciphertext="1 2 3 3 4",
            expected_ser=0.0,
            expected_smer=0.0,
            expected_exact_match=True,
        ),
        EvaluateTestCase(
            name="End-to-End Partial Match",
            raw_output="```\nHELLP\n```",
            ground_truth="HELLO",
            ciphertext="1 2 3 3 4",
            expected_ser=0.2,  # 1 char error / 5 total chars
            expected_smer=0.25,  # 1 symbol error / 4 unique symbols
            expected_exact_match=False,
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_evaluate_integration(evaluator: Evaluator, tc: EvaluateTestCase):
    """Test the orchestration of cleaning, SER, and SMER scoring."""
    result = evaluator.evaluate(tc.raw_output, tc.ground_truth, tc.ciphertext)

    assert result.raw_output == tc.raw_output
    assert result.ground_truth == tc.ground_truth

    assert result.ser == pytest.approx(tc.expected_ser, abs=1e-5)
    assert result.smer == pytest.approx(tc.expected_smer, abs=1e-5)
    assert result.is_exact_match == tc.expected_exact_match
