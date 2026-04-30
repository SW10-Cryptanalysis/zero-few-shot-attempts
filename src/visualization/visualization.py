import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logging import get_logger

logger = get_logger(__name__)


def process_csv_files(directory: str | None = None) -> None:
    """Visualizes the zero- and few-shot results from the csv files.

    Attributes:
        directory (str | None): The directory containing the CSV files to process.
                                Defaults to the 'results' folder if None.

    """
    # Fallback to default directory if none is provided
    if directory is None:
        directory = os.path.join(os.path.dirname(__file__), "..", "..", "results")

    # 1. Read all CSV files from the directory
    all_files = glob.glob(os.path.join(directory, "*.csv"))

    if not all_files:
        return

    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df["source_file"] = os.path.basename(file)
            df_list.append(df)
        except Exception:
            logger.error(f"Error reading {file}", exc_info=True)
            pass

    # Merge all data into a single DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)

    # 2. Extract the redundancy from the sample_id field
    combined_df["redundancy"] = (
        combined_df["sample_id"].str.extract(r"_R(\d+)_").astype(float)
    )

    combined_df["strategy"] = (
        combined_df["strategy"].astype(str).str.lower().str.strip()
    )
    combined_df["with_spaces"] = (
        combined_df["with_spaces"].astype(str).str.lower().str.strip()
    )

    # Ensure numeric columns are properly typed
    combined_df["ser"] = combined_df["ser"].astype(float)
    combined_df["cipher_length"] = combined_df["cipher_length"].astype(float)

    # 3. Define the 4 combinations we need to plot
    conditions = [
        ("few-shot", "true"),
        ("few-shot", "false"),
        ("zero-shot", "true"),
        ("zero-shot", "false"),
    ]

    # Set the visual style for the plots
    sns.set_theme(style="whitegrid")

    for strategy, with_spaces in conditions:
        # Filter the combined dataframe for the specific condition
        subset = combined_df[
            (combined_df["strategy"] == strategy)
            & (combined_df["with_spaces"] == with_spaces)
        ]

        if subset.empty:
            continue

        # Create a figure containing two subplots side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            f"Strategy: {strategy.title()} | With Spaces: {with_spaces.title()}",
            fontsize=16,
            fontweight="bold",
        )

        # Plot 1: SER vs Redundancy
        sns.lineplot(
            data=subset,
            x="redundancy",
            y="ser",
            hue="model",
            style="model",
            markers=True,
            dashes=False,
            ax=axes[0],
        )
        axes[0].set_title("SER vs Redundancy", fontsize=14)
        axes[0].set_xlabel("Redundancy", fontsize=12)
        axes[0].set_ylabel("SER (Symbol Error Rate)", fontsize=12)

        # Plot 2: SER vs Cipher Length
        sns.lineplot(
            data=subset,
            x="cipher_length",
            y="ser",
            hue="model",
            style="model",
            markers=True,
            dashes=False,
            ax=axes[1],
        )
        axes[1].set_title("SER vs Cipher Length", fontsize=14)
        axes[1].set_xlabel("Cipher Length", fontsize=12)
        axes[1].set_ylabel("SER (Symbol Error Rate)", fontsize=12)

        plt.tight_layout()

        # Save the graph in the results directory alongside the CSV files
        output_filename = os.path.join(
            directory,
            f"graph_{strategy}_spaces_{with_spaces}.png",
        )
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")


if __name__ == "__main__":  # pragma: no cover
    process_csv_files()
