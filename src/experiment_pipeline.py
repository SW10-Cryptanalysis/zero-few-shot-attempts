import pandas as pd
from pathlib import Path
from typing import Literal, TypedDict
from tqdm import tqdm
from src.data_handler import DataHandler, CipherSample
from src.model_client import ModelClient
from src.evaluator import Evaluator
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ResultDict(TypedDict):
    """Dictionary containing the results of an experiment."""

    sample_id: str
    model: str
    strategy: Literal["zero-shot", "few-shot"]
    ser: float
    is_exact_match: bool
    raw_output: str
    cleaned_prediction: str
    ground_truth: str
    cipher_length: int
    with_spaces: bool
    genre: str


class ExperimentPipeline:
    """Orchestrates the entire pipeline for the experiment.

    Attributes:
        handler (DataHandler): The data handler.
        client (ModelClient): The model client.
        evaluator (Evaluator): The evaluator.
        output_dir (Path): The path to the output file.

    """

    def __init__(
        self,
        handler: DataHandler,
        client: ModelClient,
        evaluator: Evaluator,
        output_dir: Path,
    ) -> None:
        """Initialize the pipeline.

        Args:
            handler (DataHandler): The data handler.
            client (ModelClient): The model client.
            evaluator (Evaluator): The evaluator.
            output_dir (Path): The path to the output file.

        """
        self.handler = handler
        self.client = client
        self.evaluator = evaluator
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_processed_ids(self, output_file: Path) -> set[str]:
        """Read the output file to see which samples are already finished.

        Args:
            output_file (Path): The path to the output file.

        Returns:
            set[str]: The set of sample IDs that have already been processed.

        """
        if not output_file.exists():
            return set()
        try:
            df = pd.read_csv(output_file, usecols=["sample_id"])
            return set(df["sample_id"].astype(str).tolist())
        except Exception as e:
            logger.warning(f"Could not read existing results for resume: {e}")
            return set()

    def run(
        self,
        batch_size: int = 10,
        strategy: Literal["zero-shot", "few-shot"] = "zero-shot",
    ) -> pd.DataFrame:
        """Run the entire pipeline.

        Args:
            batch_size (int, optional): The size of the batches to process.
                Defaults to 10.
            strategy (Literal[&quot;zero, optional): The strategy to use for the prompt.
                Defaults to "zero-shot".

        Returns:
            pd.DataFrame: The results of the experiment.

        """
        output_file = (
            self.output_dir
            / f"{self.client.config.model_name.replace('/', '_')}_{strategy}.csv"
        )

        processed_ids = self._get_processed_ids(output_file)
        if processed_ids:
            logger.info(
                f"Resuming experiment. Skipping {len(processed_ids)} already processed "
                "samples.",
            )

        logger.info(
            f"Starting {strategy} experiment for {self.client.config.model_name}",
        )

        all_results: list[ResultDict] = []

        pbar = tqdm(
            total=len(self.handler.dataset),
            desc=f"Experiment: {strategy}",
            initial=len(processed_ids),
        )

        for batch in self.handler.get_batch(batch_size):
            batch_to_process = [s for s in batch if s.sample_id not in processed_ids]

            if not batch_to_process:
                continue

            batch_results = self._process_batch(strategy, batch_to_process)

            if batch_results:
                self._append_to_csv(batch_results, output_file)
                all_results.extend(batch_results)

            pbar.update(len(batch_to_process))

        pbar.close()

        final_df = pd.read_csv(output_file)
        logger.info(
            f"Complete. Mean SER for {self.client.config.model_name}: {
                final_df['ser'].mean():.4f}",
        )
        return final_df

    def _process_batch(
        self,
        strategy: Literal["zero-shot", "few-shot"],
        batch: list[CipherSample],
    ) -> list[ResultDict]:
        batch_results: list[ResultDict] = []

        for sample in batch:
            try:
                messages = self.handler.format_prompt(sample, strategy)
                response = self.client.generate_response(messages)

                if not response:
                    logger.warning(
                        f"Empty response for sample {sample.sample_id}. Skipping.",
                    )
                    continue

                result = self.evaluator.evaluate(response, sample.plaintext)

                result_entry: ResultDict = {
                    "sample_id": sample.sample_id,
                    "model": self.client.config.model_name,
                    "strategy": strategy,
                    "ser": result.ser,
                    "is_exact_match": result.is_exact_match,
                    "raw_output": result.raw_output,
                    "cleaned_prediction": result.cleaned_prediction,
                    "ground_truth": sample.plaintext,
                    "cipher_length": sample.metadata.length,
                    "with_spaces": sample.metadata.with_spaces,
                    "genre": sample.metadata.genre,
                }
                batch_results.append(result_entry)
            except Exception as e:
                logger.error(
                    f"Critical error processing sample {sample.sample_id}: {e}",
                )
                continue

        return batch_results

    @staticmethod
    def _append_to_csv(results: list[ResultDict], output_file: Path) -> None:
        df = pd.DataFrame(results)
        header = not output_file.exists()
        df.to_csv(output_file, mode="a", header=header, index=False)
