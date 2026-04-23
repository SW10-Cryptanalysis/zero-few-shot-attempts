import re
import editdistance
from dataclasses import dataclass
from collections import defaultdict, Counter


@dataclass
class EvaluationResult:
    """Result of an evaluation.

    Attributes:
        ser (float): The edit distance between the prediction and ground truth.
        prediction (str): The prediction from the model.
        ground_truth (str): The ground truth plaintext.

    """

    ser: float
    smer: float
    raw_output: str
    cleaned_prediction: str
    ground_truth: str
    is_exact_match: bool


class Evaluator:
    """Evaluator for the ciphers."""

    TAU = 0.95

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

    def _extract_keys(
        self,
        prediction: str,
        ground_truth: str,
        ciphertext: str,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Extract the key from the prediction.

        Args:
            prediction (str): The prediction from the model.
            ground_truth (str): The ground truth plaintext.
            ciphertext (str): The ciphertext.

        Returns:
            tuple[dict[str, str], dict[str, dict[str, int]]]: The key and predicted key.

        Raises:
            ValueError: If the ground truth does not respect 1:many mapping.

        """
        extracted_true_key = {}
        mapping_freqs = defaultdict(Counter)

        for idx, symbol in enumerate(ciphertext.split()):
            if symbol in extracted_true_key:
                if not extracted_true_key[symbol] == ground_truth[idx]:
                    raise ValueError(
                        f"Ground truth and extracted key do not match at index {idx}: "
                        f"\nPredicted letter: {prediction[idx]}"
                        f"\nGround truth letter: {ground_truth[idx]}"
                        f"\nCiphertext symbol: {symbol}"
                        f"\nKey: {extracted_true_key}",
                    )
            else:
                extracted_true_key[symbol] = ground_truth[idx]
            if idx < len(prediction):
                mapping_freqs[symbol][prediction[idx]] += 1

        predicted_key = self._get_emprirical_key(mapping_freqs)

        return extracted_true_key, predicted_key

    def _get_emprirical_key(
        self,
        mapping_freqs: dict[str, Counter],
    ) -> dict[str, str]:
        """Get the empirical key from the mapping frequencies.

        Args:
            mapping_freqs (dict[str, dict[str, int]]): The mapping frequencies.

        Returns:
            dict[str, str]: The empirical key.

        """
        empirical_key = {}
        for symbol, counts in mapping_freqs.items():
            total_count = sum(counts.values())
            if total_count == 0:
                continue

            predicted_letter = max(counts, key=lambda k: counts[k])
            letter_count = counts[predicted_letter]

            if letter_count / total_count >= self.TAU:
                empirical_key[symbol] = predicted_letter

        return empirical_key

    def _calculate_smer(
        self,
        prediction: str,
        ground_truth: str,
        ciphertext: str,
    ) -> float:
        """Calculate the Strict Mapping Error Rate.

        Args:
            prediction (str): The prediction from the model.
            ground_truth (str): The ground truth plaintext.
            ciphertext (str): The ciphertext.


        Returns:
            float: The SMER between the prediction and ground truth.

        """
        if not prediction or not ground_truth:
            return 1.0

        key, predicted_key = self._extract_keys(prediction, ground_truth, ciphertext)

        mismatches = 0
        for symbol in key:
            if symbol not in predicted_key or key[symbol] != predicted_key[symbol]:
                mismatches += 1

        return mismatches / len(key)

    def evaluate(
        self,
        raw_output: str,
        ground_truth: str,
        ciphertext: str,
    ) -> EvaluationResult:
        """Evaluate the model's output against the ground truth.

        Args:
            raw_output (str): The raw output from the model.
            ground_truth (str): The ground truth plaintext.
            ciphertext (str): The ciphertext.

        Returns:
            EvaluationResult: The evaluation result.

        """
        prediction = self._clean_prediction(raw_output)
        ser = self._calculate_ser(prediction, ground_truth)
        smer = self._calculate_smer(prediction, ground_truth, ciphertext)
        return EvaluationResult(
            ser=ser,
            smer=smer,
            raw_output=raw_output,
            cleaned_prediction=prediction,
            ground_truth=ground_truth,
            is_exact_match=(ser == 0.0),
        )
