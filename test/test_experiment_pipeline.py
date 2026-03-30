import pytest
import pandas as pd
from dataclasses import dataclass
from src.data_handler import CipherSample, CipherMetadata


def create_full_dummy_result(sample_id: str) -> dict:
    """Helper to create a result dict with the full ResultDict schema."""
    return {
        "sample_id": sample_id,
        "model": "test-model",
        "strategy": "zero-shot",
        "ser": 0.0,
        "is_exact_match": True,
        "raw_output": "...",
        "cleaned_prediction": "...",
        "ground_truth": "...",
        "cipher_length": 10,
        "with_spaces": False,
        "genre": "test",
    }


def test_pipeline_initialization(pipeline):
    """Verify that the pipeline creates the output directory on init."""
    assert pipeline.output_dir.exists()
    assert pipeline.output_dir.is_dir()


@dataclass
class ProcessedIdsTestCase:
    """Test case data for testing _get_processed_ids logic."""

    name: str
    file_exists: bool
    is_corrupt: bool
    expected_ids: set[str]


@pytest.mark.parametrize(
    "tc",
    [
        ProcessedIdsTestCase("Existing valid file", True, False, {"id_1", "id_2"}),
        ProcessedIdsTestCase("Missing file", False, False, set()),
        ProcessedIdsTestCase("Corrupted file", True, True, set()),
    ],
    ids=lambda tc: tc.name,
)
def test_get_processed_ids(pipeline, tmp_path, mocker, tc: ProcessedIdsTestCase):
    """Test the _get_processed_ids method across varying file states."""
    output_file = pipeline.output_dir / f"test_{tc.name.replace(' ', '_')}.csv"

    if tc.file_exists:
        if tc.is_corrupt:
            output_file.touch()
            mocker.patch(
                "src.experiment_pipeline.pd.read_csv",
                side_effect=Exception("Corrupt CSV"),
            )
        else:
            df = pd.DataFrame(
                [create_full_dummy_result("id_1"), create_full_dummy_result("id_2")]
            )
            df.to_csv(output_file, index=False)

    processed_ids = pipeline._get_processed_ids(output_file)
    assert processed_ids == tc.expected_ids


@dataclass
class RunOrchestrationTestCase:
    """Test case data for testing the run loop and batch skipping."""

    name: str
    pre_existing_ids: list[str]
    batch_ids: list[str]
    expected_api_calls: int


@pytest.mark.parametrize(
    "tc",
    [
        RunOrchestrationTestCase("Fresh run", [], ["id_1", "id_2"], 2),
        RunOrchestrationTestCase("Partial resume", ["id_1"], ["id_1", "id_2"], 1),
        RunOrchestrationTestCase(
            "Complete batch skip", ["id_1", "id_2"], ["id_1", "id_2"], 0
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_run_orchestration(pipeline, mock_components, tc: RunOrchestrationTestCase):
    """Test the run method logic for skipping processed samples and calling the client."""
    handler, client, _ = mock_components
    output_file = pipeline.output_dir / "test-provider_test-model_zero-shot.csv"

    if tc.pre_existing_ids:
        df = pd.DataFrame(
            [create_full_dummy_result(sid) for sid in tc.pre_existing_ids]
        )
        df.to_csv(output_file, index=False)

    samples = [
        CipherSample(sid, "P", "C", {}, CipherMetadata(1, False, "t"))
        for sid in tc.batch_ids
    ]
    handler.get_batch.return_value = [samples]
    handler.dataset = samples

    pipeline.run(batch_size=10, strategy="zero-shot")

    assert client.generate_response.call_count == tc.expected_api_calls


@dataclass
class ProcessBatchTestCase:
    """Test case data for testing error resiliency in _process_batch."""

    name: str
    client_effects: list[str]
    eval_crashes_on_first: bool
    sample_ids: list[str]
    expected_success_ids: list[str]


@pytest.mark.parametrize(
    "tc",
    [
        ProcessBatchTestCase(
            "Client returns empty response",
            ["SUCCESS_VAL", ""],
            False,
            ["good", "bad"],
            ["good"],
        ),
        ProcessBatchTestCase(
            "Evaluator raises exception",
            ["SUCCESS_VAL", "SUCCESS_VAL"],
            True,
            ["crash_me", "survive_me"],
            ["survive_me"],
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_process_batch_errors(
    pipeline, mock_components, mocker, tc: ProcessBatchTestCase
):
    """Test that _process_batch isolates failures and continues processing."""
    handler, client, evaluator = mock_components

    client.generate_response.side_effect = tc.client_effects

    if tc.eval_crashes_on_first:
        evaluator.evaluate.side_effect = [Exception("Random Crash"), mocker.Mock()]

    samples = [
        CipherSample(sid, "P", "C", {}, CipherMetadata(1, False, "t"))
        for sid in tc.sample_ids
    ]

    results = pipeline._process_batch("zero-shot", samples)

    assert len(results) == len(tc.expected_success_ids)
    for result_entry, expected_id in zip(results, tc.expected_success_ids, strict=True):
        assert result_entry["sample_id"] == expected_id
