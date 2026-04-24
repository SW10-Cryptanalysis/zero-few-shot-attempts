import pandas as pd
from src.utils.logging import get_logger
import argparse
import os

logger = get_logger(__name__)


def mean_ser(
    model: str,
    strategy: str = "zero-shot",
) -> tuple[float, float, float, int]:
    """Load the results from the CSV and calculate the mean SER.

    Args:
        model (str): The model name.
        strategy (str, optional): The strategy to use. Defaults to "zero-shot".

    Returns:
        float: The mean SER.

    """
    files = list(
        filter(lambda x: model in x and strategy in x, os.listdir("data/results")),
    )
    if not files:
        raise FileNotFoundError(f"No results found for model {model}")

    results = pd.read_csv(f"data/results/{files[0]}")
    mean_ser = results["ser"].mean()
    homophonic_results = results[~results["sample_id"].str.contains("_R0_")]
    ser_homophonic = homophonic_results["ser"].mean()
    mono_results = results[results["sample_id"].str.contains("_R0_")]
    ser_mono = mono_results["ser"].mean()
    num_samples = results["sample_id"].nunique()

    return mean_ser, ser_homophonic, ser_mono, num_samples


def retrieve_sers(
    model: str,
    strategy: str = "zero-shot",
) -> list[tuple[str, str, float, float, float, int]]:
    """Load the results from the CSV and retrieve the SERs.

    Args:
        model (str): The model name.
        strategy (str, optional): The strategy to use. Defaults to "zero-shot".

    Returns:
        list[tuple[str, str, float]]: Tuples of (model, strategy, ser).

    """
    mean_sers: list[tuple[str, str, float, float, float, int]] = []
    for model in os.listdir("data/results"):
        try:
            strategy = model.split("_")[2].replace(".csv", "")
        except IndexError:
            logger.error(f"No strategy found in {model}")
            continue

        model = model.split("_")[1]
        if not model:
            logger.error(f"No model name found in {model}")
            continue
        ser, ser_homophonic, ser_mono, num_samples = mean_ser(
            model=model,
            strategy=strategy,
        )
        mean_sers.append(
            (
                model,
                strategy,
                ser,
                ser_homophonic,
                ser_mono,
                num_samples,
            ),
        )
    return mean_sers


def format_table(mean_sers: list[tuple[str, str, float, float, float, int]]) -> str:
    """Format the mean SERs as a table.

    Args:
        mean_sers (list[tuple[str, str, float]]): The mean SERs.

    Returns:
        str: The formatted table.

    """
    mean_sers.sort(key=lambda x: x[2])
    header = "{:<30} | {:<10} | {:>8} | {:>14} | {:>14} | {:>7}".format(
        "Model",
        "Strategy",
        "Mean SER",
        "Homo. Mean SER",
        "Mono. Mean SER",
        "Samples",
    )
    line = (
        "-" * 31
        + "+"
        + "-" * 12
        + "+"
        + "-" * 10
        + "+"
        + "-" * 16
        + "+"
        + "-" * 16
        + "+"
        + "-" * 9
    )
    return "\n".join(
        [
            "Mean SER by Model and Strategy:",
            header,
            line,
            *[
                f"{model:<30} | {strategy:<10} | {ser:>8.4f} | {ser_homophonic:>14.4f}"
                f" | {ser_mono:>14.4f} | {num_samples:>7}"
                for (
                    model,
                    strategy,
                    ser,
                    ser_homophonic,
                    ser_mono,
                    num_samples,
                ) in mean_sers
            ],
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--strategy", type=str, default="zero-shot")
    args = parser.parse_args()
    if not args.model:
        try:
            mean_sers = retrieve_sers(model=args.model, strategy=args.strategy)
            logger.info(format_table(mean_sers))
            exit(0)
        except Exception as e:
            logger.error(e)
            exit(1)

    try:
        ser, mono_ser, ser_homophonic, num_samples = mean_ser(
            model=args.model,
            strategy=args.strategy,
        )
        logger.info(
            f"Mean SER for {args.model} ({args.strategy}): {ser:.4f}, (mono "
            f"{mono_ser:.4f}, homophonic {ser_homophonic:.4f})",
        )
    except FileNotFoundError as e:
        logger.error(e)
        exit(1)
