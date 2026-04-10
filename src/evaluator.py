import re
import editdistance
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Result of an evaluation.

    Attributes:
        ser (float): The edit distance between the prediction and ground truth.
        prediction (str): The prediction from the model.
        ground_truth (str): The ground truth plaintext.

    """

    ser: float
    raw_output: str
    cleaned_prediction: str
    ground_truth: str
    is_exact_match: bool


class Evaluator:
    """Evaluator for the ciphers."""

    def __init__(self) -> None:
        """Initialize the evaluator."""
        pass

    def _clean_prediction(self, raw_output: str) -> str:
        """Clean the prediction by removing extraneous characters.

        Args:
            raw_output (str): The raw output from the model.

        Returns:
            str: The cleaned prediction.

        """
        if not raw_output:
            return ""

        cleaned = re.sub(
            r"```[a-zA-Z]*\n(.*?)\n```",
            r"\1",
            raw_output,
            flags=re.DOTALL,
        )

        longest_word = max(cleaned.split(), key=len)
        return longest_word.strip().replace(",", "")

    def _calculate_ser(self, prediction: str, ground_truth: str) -> float:
        """Calculate the edit distance between the prediction and ground truth.

        Args:
            prediction (str): The prediction from the model.
            ground_truth (str): The ground truth plaintext.

        Returns:
            float: The edit distance between the prediction and ground truth.

        """
        if not prediction or not ground_truth:
            return 1.0

        distance = editdistance.eval(prediction, ground_truth)

        ser = distance / len(ground_truth)

        return min(ser, 1.0)

    def evaluate(self, raw_output: str, ground_truth: str) -> EvaluationResult:
        """Evaluate the model's output against the ground truth.

        Args:
            raw_output (str): The raw output from the model.
            ground_truth (str): The ground truth plaintext.

        Returns:
            float: The edit distance between the prediction and ground truth.

        """
        prediction = self._clean_prediction(raw_output)
        ser = self._calculate_ser(prediction, ground_truth)
        return EvaluationResult(
            ser=ser,
            raw_output=raw_output,
            cleaned_prediction=prediction,
            ground_truth=ground_truth,
            is_exact_match=(ser == 0.0),
        )
