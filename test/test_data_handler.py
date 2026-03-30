import json
import pytest
from pathlib import Path
from dataclasses import dataclass
from typing import Literal
from src.data_handler import DataHandler, CipherSample, CipherMetadata

# =====================================================================
# 1. Test Helpers & Fixtures
# =====================================================================


def generate_jsonl(file_path: Path, num_records: int) -> None:
    """Helper to generate a real, temporary JSONL file for testing."""
    lines = []
    for i in range(num_records):
        record = {
            "id": f"id_{i}",
            "ciphertext": f"abc_{i}",
            "plaintext": f"def_{i}",
            "key": {"a": ["b", "c"]},
            "genres": "test_genre",
        }
        lines.append(json.dumps(record))
    file_path.write_text("\n".join(lines), encoding="utf-8")


def create_dummy_sample(id_str: str) -> CipherSample:
    """Helper to generate dummy CipherSamples in memory."""
    return CipherSample(
        sample_id=id_str,
        ciphertext=f"cipher_{id_str}",
        plaintext=f"plain_{id_str}",
        key={"x": ["y"]},
        metadata=CipherMetadata(length=10, with_spaces=False, genre="dummy"),
    )


@pytest.fixture
def handler_setup(tmp_path: Path) -> DataHandler:
    """Fixture providing a configured DataHandler with valid dummy paths."""
    data_path = tmp_path / "data.jsonl"
    few_shot_path = tmp_path / "few_shot.jsonl"

    # Touch files so they exist, even if empty
    data_path.touch()
    few_shot_path.touch()

    return DataHandler(
        data_path=data_path,
        few_shot_data_path=few_shot_path,
        system_prompt="You are a cryptanalyst.",
    )


# =====================================================================
# 2. Dataclasses for Parametrized Tests
# =====================================================================


@dataclass
class BatchTestCase:
    name: str
    total_samples: int
    batch_size: int
    expected_batches: int
    expected_last_batch_size: int


@dataclass
class PromptTestCase:
    name: str
    strategy: Literal["zero-shot", "few-shot"]
    num_few_shot_examples: int
    expected_roles: list[str]


# =====================================================================
# 3. Unit Tests
# =====================================================================


def test_initialization(handler_setup: DataHandler):
    """Test that attributes are correctly assigned on initialization."""
    assert handler_setup.system_prompt == "You are a cryptanalyst."
    assert isinstance(handler_setup.dataset, list)
    assert len(handler_setup.dataset) == 0
    assert len(handler_setup.few_shot_examples) == 0


def test_parse_file(handler_setup: DataHandler, tmp_path: Path):
    """Test file parsing using real temporary files instead of mock_open."""
    test_file = tmp_path / "test_parse.jsonl"
    generate_jsonl(test_file, num_records=3)

    parsed_data = handler_setup._parse_file(test_file, with_spaces=True)

    assert len(parsed_data) == 3
    assert parsed_data[0].sample_id == "id_0"
    assert parsed_data[0].metadata.with_spaces is True
    assert parsed_data[0].metadata.genre == "test_genre"
    assert parsed_data[0].metadata.length == len("abc_0")


def test_load_data_calls_parse_correctly(handler_setup: DataHandler, mocker):
    """
    Test load_data orchestrates _parse_file correctly.
    Uses mocker.spy to track calls without overriding the method with a MagicMock.
    """
    generate_jsonl(handler_setup.data_path, num_records=5)
    generate_jsonl(handler_setup.few_shot_data_path, num_records=2)

    # Spy on the method to verify behavior
    spy_parse = mocker.spy(handler_setup, "_parse_file")

    handler_setup.load_data(with_spaces=False)

    # Verify state changes
    assert len(handler_setup.dataset) == 5
    assert len(handler_setup.few_shot_examples) == 2

    # Verify correct orchestration
    assert spy_parse.call_count == 2
    spy_parse.assert_any_call(handler_setup.data_path, False)
    spy_parse.assert_any_call(handler_setup.few_shot_data_path, False)


@pytest.mark.parametrize(
    "tc",
    [
        BatchTestCase(
            name="Perfect division",
            total_samples=10,
            batch_size=5,
            expected_batches=2,
            expected_last_batch_size=5,
        ),
        BatchTestCase(
            name="Remainder",
            total_samples=10,
            batch_size=3,
            expected_batches=4,
            expected_last_batch_size=1,
        ),
        BatchTestCase(
            name="Larger than dataset",
            total_samples=5,
            batch_size=10,
            expected_batches=1,
            expected_last_batch_size=5,
        ),
        BatchTestCase(
            name="Empty dataset",
            total_samples=0,
            batch_size=5,
            expected_batches=0,
            expected_last_batch_size=0,
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_get_batch(handler_setup: DataHandler, tc: BatchTestCase):
    """Test that get_batch correctly chunks the dataset."""
    # Artificially populate the dataset for this test
    handler_setup.dataset = [
        create_dummy_sample(str(i)) for i in range(tc.total_samples)
    ]

    batches = list(handler_setup.get_batch(tc.batch_size))

    assert len(batches) == tc.expected_batches
    if tc.expected_batches > 0:
        assert len(batches[-1]) == tc.expected_last_batch_size


@pytest.mark.parametrize(
    "tc",
    [
        PromptTestCase(
            name="Zero Shot Strategy",
            strategy="zero-shot",
            num_few_shot_examples=3,  # Should be ignored
            expected_roles=["system", "user"],
        ),
        PromptTestCase(
            name="Few Shot Strategy (1 example)",
            strategy="few-shot",
            num_few_shot_examples=1,
            expected_roles=["system", "user", "assistant", "user"],
        ),
        PromptTestCase(
            name="Few Shot Strategy (3 examples)",
            strategy="few-shot",
            num_few_shot_examples=3,
            expected_roles=[
                "system",
                "user",
                "assistant",  # FS 1
                "user",
                "assistant",  # FS 2
                "user",
                "assistant",  # FS 3
                "user",  # Actual Prompt
            ],
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_format_prompt(handler_setup: DataHandler, tc: PromptTestCase):
    """Test that message arrays are formatted with the correct roles and order."""

    # Populate few shot examples if needed
    handler_setup.few_shot_examples = [
        create_dummy_sample(f"fs_{i}") for i in range(tc.num_few_shot_examples)
    ]

    target_sample = create_dummy_sample("target")

    messages = handler_setup.format_prompt(sample=target_sample, strategy=tc.strategy)

    # Extract just the roles to verify structure
    actual_roles = [msg["role"] for msg in messages]

    assert actual_roles == tc.expected_roles
    assert messages[0]["content"] == handler_setup.system_prompt
    assert target_sample.ciphertext in messages[-1]["content"]
