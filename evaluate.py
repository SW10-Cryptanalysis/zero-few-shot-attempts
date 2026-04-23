from pathlib import Path
from src.evaluator import Evaluator
from src.utils.logging import get_logger
import json
import pandas as pd
from src.experiment_pipeline import ResultDict

MODEL_NAME = "openai/gpt-5.4-thinking"
STRATEGY = "few-shot"

logger = get_logger(__name__)


def load_outputs(output_dir: Path) -> list[dict[str, str]]:
    """Load the ouput/ground truth pairs from data/manual_outputs.jsonl."""
    samples = []
    outputs = []
    with open(output_dir / "manual_outputs.jsonl", encoding="utf-8") as f:
        for line in f:
            outputs.append(json.loads(line))
    with open(output_dir / "manual_dataset.jsonl", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            samples.append(sample)
    for i, _ in enumerate(samples):
        samples[i]["output"] = outputs[i]["output"]

    return samples


def main() -> None:
    """Start the application."""
    evaluator = Evaluator()
    data_dir = Path("data")
    samples = load_outputs(data_dir)
    results = []

    for sample in samples:
        result = evaluator.evaluate(
            sample["output"],
            sample["plaintext"],
            sample["ciphertext"],
        )
        if not result.raw_output:
            logger.error(f"Raw output is None for sample {sample['id']}")
        result_obj: ResultDict = {
            "sample_id": sample["id"],
            "model": MODEL_NAME,
            "strategy": STRATEGY,
            "ser": result.ser,
            "smer": result.smer,
            "is_exact_match": result.is_exact_match,
            "raw_output": result.raw_output,
            "cleaned_prediction": result.cleaned_prediction,
            "ground_truth": sample["plaintext"],
            "cipher_length": int(sample["length"]),
            "with_spaces": False,
            "genre": sample["genres"],
        }
        results.append(result_obj)

    # Save the results to a CSV file
    data_frame = pd.DataFrame(results)
    data_frame.to_csv(
        data_dir / "results" / f"{MODEL_NAME.replace('/', '_')}_{STRATEGY}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
