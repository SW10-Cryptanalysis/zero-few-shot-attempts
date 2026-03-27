from typing import TypedDict, Literal, Iterator
from dataclasses import dataclass
from pathlib import Path
import json


class APIMessage(TypedDict):
    """Formatted message for the API."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class CipherMetadata:
    """Metadata for a cipher."""

    length: int
    with_spaces: bool
    genre: str


@dataclass
class CipherSample:
    """A sample of a ciphertext-plaintext pair and its key."""

    sample_id: str
    plaintext: str
    ciphertext: str
    key: dict[str, list[str]]
    metadata: CipherMetadata


class DataHandler:
    """Data handler for the ciphers.

    Attributes:
        data_path (Path): The path to the data.
        dataset (list[CipherSample]): The dataset of cipher samples.

    """

    def __init__(
        self,
        data_path: Path,
        few_shot_data_path: Path,
        system_prompt: str,
    ) -> None:
        """Initialize the data handler.

        Args:
            data_path (Path): The path to the data.
            few_shot_data_path (Path): The path to the few-shot data.
            system_prompt (str): The system prompt to use for the model.

        """
        self.data_path = data_path
        self.few_shot_data_path = few_shot_data_path
        self.dataset: list[CipherSample] = []
        self.few_shot_examples: list[CipherSample] = []
        self.system_prompt = system_prompt

    def load_data(self, with_spaces: bool = False) -> None:
        """Load the data from the data path.

        Args:
            with_spaces (bool, optional): Whether to include spaces in the ciphertext.
                Defaults to False.

        """
        self.dataset = self._parse_file(self.data_path, with_spaces)
        self.few_shot_examples = self._parse_file(self.few_shot_data_path, with_spaces)

    def _parse_file(self, path: Path, with_spaces: bool) -> list[CipherSample]:
        """Private helper to parse JSONL files into CipherSample objects."""
        parsed_data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                sample = CipherSample(
                    sample_id=row["id"],
                    ciphertext=row["cipher"],
                    plaintext=row["plaintext"],
                    key=row["key"],
                    metadata=CipherMetadata(
                        length=len(row["cipher"]),
                        with_spaces=with_spaces,
                        genre=row["genre"],
                    ),
                )
                parsed_data.append(sample)
        return parsed_data

    def get_batch(self, batch_size: int) -> Iterator[list[CipherSample]]:
        """Get a batch of data from the data loader.

        Args:
            batch_size (int): The size of the batch to return.

        Returns:
            list[CipherData]: A list of size `batch_size` containing cipher data.

        """
        for i in range(0, len(self.dataset), batch_size):
            yield self.dataset[i : i + batch_size]

    def format_prompt(
        self,
        sample: CipherSample,
        strategy: Literal["zero-shot", "few-shot"] = "zero-shot",
    ) -> list[APIMessage]:
        """Format a prompt for the model.

        Args:
            sample (CipherSample): The sample to format the prompt for.
            strategy (Literal["zero-shot", "few-shot"]): The strategy to use for the
                prompt.

        Returns:
            list[ApiMessage]: The formatted prompt.

        """
        messages: list[APIMessage] = []

        system_msg: APIMessage = {
            "role": "system",
            "content": self.system_prompt,
        }
        messages.append(system_msg)

        if strategy == "few-shot":
            for fs_sample in self.few_shot_examples:
                user_msg: APIMessage = {
                    "role": "user",
                    "content": f"Ciphertext: {fs_sample.ciphertext}",
                }

                messages.append(user_msg)

                answer_msg: APIMessage = {
                    "role": "assistant",
                    "content": f"{fs_sample.plaintext}",
                }

                messages.append(answer_msg)

        user_msg: APIMessage = {
            "role": "user",
            "content": f"Ciphertext: {sample.ciphertext}",
        }

        messages.append(user_msg)

        return messages
