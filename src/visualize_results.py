import argparse
import csv
import glob
import logging
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
from easy_logging import EasyFormatter

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def extract_redundancy(sample_id: str) -> int | None:
    """Extract the integer following '_R' in the sample_id."""
    if not sample_id:
        return None
    match = re.search(r"_R(\d+)", sample_id)
    return int(match.group(1)) if match else None

def parse_csv_files(csv_files: list[str]) -> dict:
    """Parse data from all discovered CSV files and group them."""
    data_by_model = {}

    for file_path in csv_files:
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row.get("model")
                strategy = row.get("strategy")
                sample_id = row.get("sample_id")
                ser_str = row.get("ser")
                length_str = row.get("cipher_length")

                if not (model and strategy and ser_str and length_str):
                    logger.warning(f"Missing field for {file_path},"
                                   f"sample {sample_id}. Skipping sample...")
                    continue

                redundancy = extract_redundancy(sample_id)
                if redundancy is None:
                    logger.warning(f"Could not extract redundancy for {file_path},"
                                   f"sample {sample_id}. Skipping sample...")
                    continue

                try:
                    ser = float(ser_str)
                    length = int(length_str)
                except ValueError:
                    continue

                model = model.strip()
                strategy = strategy.strip()

                if model not in data_by_model:
                    data_by_model[model] = {}
                if strategy not in data_by_model[model]:
                    data_by_model[model][strategy] = {
                        "lengths": [],
                        "redundancies": [],
                        "sers": [],
                    }

                data_by_model[model][strategy]["lengths"].append(length)
                data_by_model[model][strategy]["redundancies"].append(redundancy)
                data_by_model[model][strategy]["sers"].append(ser)

    return data_by_model


def generate_model_plots(model_name: str, strategies: dict, output_dir: Path) -> None:
    """Generate and save the dual scatter plots for a specific model."""
    strategy_colors = {
        "zero-shot": "#1f77b4",
        "few-shot": "#d62728",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"{model_name}",
        fontsize=15,
        fontweight="bold",
    )

    for strategy_name, metrics in strategies.items():
        color = strategy_colors.get(strategy_name.lower())

        ax1.scatter(
            metrics["lengths"],
            metrics["sers"],
            alpha=0.6,
            color=color,
            edgecolors="none",
            label=strategy_name,
        )

        ax2.scatter(
            metrics["redundancies"],
            metrics["sers"],
            alpha=0.6,
            color=color,
            edgecolors="none",
            label=strategy_name,
        )

    ax1.set_title("Symbol Error Rate (SER) vs Cipher Length")
    ax1.set_xlabel("Cipher Length")
    ax1.set_ylabel("SER")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(title="Strategy")

    ax2.set_title("Symbol Error Rate (SER) vs Redundancy")
    ax2.set_xlabel("Redundancy")
    ax2.set_ylabel("SER")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(title="Strategy")

    plt.tight_layout()

    clean_filename = re.sub(r'[\\/*?:"<>| ]', "_", model_name).lower()
    output_image_path = output_dir / f"{clean_filename}.png"

    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Graphs successfully saved for {model_name} to:\n{output_image_path}")


def main() -> None:
    """Create dual scatter plots per model."""
    csv.field_size_limit(1024 * 1024)
    parser = argparse.ArgumentParser(
        description="Visualize evaluation results from standard CSV files.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory containing the evaluation CSV files",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists() or not results_dir.is_dir():
        logger.info(f"Error: Directory not found at {results_dir}")
        return

    output_dir = Path("graphs")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        logger.info(f"No CSV files found in {results_dir}")
        return

    data_by_model = parse_csv_files(csv_files)

    if not data_by_model:
        logger.info("No valid sample data parsed from CSV files.")
        return

    for model_name, strategies in data_by_model.items():
        generate_model_plots(model_name, strategies, output_dir)


if __name__ == "__main__":
    main()
