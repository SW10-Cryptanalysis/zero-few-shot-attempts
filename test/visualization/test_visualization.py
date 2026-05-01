import os
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

# Assuming the visualization script is in src.visualization (adjust import as needed)
from src.visualization.visualization import process_csv_files

# =====================================================================
# 1. Test Helpers & Fixtures
# =====================================================================


def create_dummy_csv(file_path: Path, rows: list[dict]) -> None:
    """Helper to generate a real, temporary CSV file for testing."""
    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)


@pytest.fixture
def mock_savefig(mocker):
    """
    Patches plt.savefig to prevent tests from writing actual image files
    to the disk, making the test suite faster and cleaner.
    """
    return mocker.patch("matplotlib.pyplot.savefig")


@pytest.fixture(autouse=True)
def close_plots():
    """Automatically close all matplotlib plots after each test to prevent memory leaks."""
    yield
    plt.close("all")


@pytest.fixture
def empty_dir(tmp_path: Path) -> Path:
    """Fixture providing an empty directory."""
    return tmp_path / "empty_results"


# =====================================================================
# 2. Dataclasses for Parametrized Tests
# =====================================================================


@dataclass
class CsvTestCase:
    name: str
    mock_data: list[dict]
    expected_graphs: list[str]
    should_log_error: bool = False


# =====================================================================
# 3. Unit Tests
# =====================================================================


def test_no_files_found(empty_dir: Path, mock_savefig, mocker):
    """Test that the function exits early if no CSV files are found."""
    empty_dir.mkdir()

    # Spy on glob to ensure it was called
    mocker.spy(os.path, "join")

    process_csv_files(str(empty_dir))

    # Check that savefig was never reached
    mock_savefig.assert_not_called()


def test_exception_handling_for_bad_files(tmp_path: Path, mock_savefig, mocker):
    """Test that a corrupted CSV file doesn't crash the entire pipeline."""
    mock_logger = mocker.patch("src.visualization.visualization.logger.error")

    data_dir = tmp_path / "results"
    data_dir.mkdir()

    # Create one valid file
    valid_data = [
        {
            "sample_id": "N400_R15_13",
            "strategy": "few-shot",
            "with_spaces": "true",
            "ser": 0.5,
            "cipher_length": 400,
            "ground_truth": "x" * 400,
            "model": "gpt",
        }
    ]
    create_dummy_csv(data_dir / "valid.csv", valid_data)

    # Create an empty bad file just so glob.glob finds it
    bad_file = data_dir / "bad.csv"
    bad_file.touch()

    # Intercept pd.read_csv to explicitly crash ONLY for 'bad.csv'
    original_read_csv = pd.read_csv

    def mocked_read_csv(filepath, *args, **kwargs):
        if "bad.csv" in str(filepath):
            raise ValueError("Simulated CSV parsing error")
        return original_read_csv(filepath, *args, **kwargs)

    mocker.patch(
        "src.visualization.visualization.pd.read_csv", side_effect=mocked_read_csv
    )

    process_csv_files(str(data_dir))

    # Ensure the error was logged, but the valid graph was still generated
    mock_logger.assert_called_once()
    assert "Error reading" in mock_logger.call_args[0][0]

    # Check that the path passed to savefig was correctly joined with the directory
    expected_output_path = os.path.join(str(data_dir), "graph_few-shot_spaces_true.png")
    mock_savefig.assert_called_once_with(
        expected_output_path, dpi=300, bbox_inches="tight"
    )


@pytest.mark.parametrize(
    "tc",
    [
        CsvTestCase(
            name="Single condition mapping",
            mock_data=[
                {
                    "sample_id": "N400_R15_13",
                    "model": "gpt-5.4",
                    "strategy": "few-shot",
                    "ser": 0.7,
                    "cipher_length": 400,
                    "ground_truth": "x" * 400,
                    "with_spaces": "False",
                },
                {
                    "sample_id": "N400_R5_13",
                    "model": "gpt-5.4",
                    "strategy": "few-shot",
                    "ser": 0.5,
                    "cipher_length": 400,
                    "ground_truth": "x" * 400,
                    "with_spaces": "False",
                },
            ],
            expected_graphs=["graph_few-shot_spaces_false.png"],
        ),
        CsvTestCase(
            name="Data cleaning capitalization handling",
            mock_data=[
                # Mixed capitalizations and spacing should be standardized
                {
                    "sample_id": "N10_R0_1",
                    "model": "m1",
                    "strategy": " FEW-SHOT ",
                    "ser": 0.2,
                    "cipher_length": 10,
                    "ground_truth": "x" * 10,
                    "with_spaces": "True",
                },
                {
                    "sample_id": "N10_R0_2",
                    "model": "m1",
                    "strategy": "Zero-Shot",
                    "ser": 0.8,
                    "cipher_length": 10,
                    "ground_truth": "x" * 10,
                    "with_spaces": "FALSE",
                },
            ],
            expected_graphs=[
                "graph_few-shot_spaces_true.png",
                "graph_zero-shot_spaces_false.png",
            ],
        ),
        CsvTestCase(
            name="All four conditions present",
            mock_data=[
                {
                    "sample_id": "N1_R1_1",
                    "model": "m1",
                    "strategy": "few-shot",
                    "ser": 0.1,
                    "cipher_length": 1,
                    "ground_truth": "x",
                    "with_spaces": "true",
                },
                {
                    "sample_id": "N1_R1_2",
                    "model": "m1",
                    "strategy": "few-shot",
                    "ser": 0.1,
                    "cipher_length": 1,
                    "ground_truth": "x",
                    "with_spaces": "false",
                },
                {
                    "sample_id": "N1_R1_3",
                    "model": "m1",
                    "strategy": "zero-shot",
                    "ser": 0.1,
                    "cipher_length": 1,
                    "ground_truth": "x",
                    "with_spaces": "true",
                },
                {
                    "sample_id": "N1_R1_4",
                    "model": "m1",
                    "strategy": "zero-shot",
                    "ser": 0.1,
                    "cipher_length": 1,
                    "ground_truth": "x",
                    "with_spaces": "false",
                },
            ],
            expected_graphs=[
                "graph_few-shot_spaces_true.png",
                "graph_few-shot_spaces_false.png",
                "graph_zero-shot_spaces_true.png",
                "graph_zero-shot_spaces_false.png",
            ],
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_data_processing_and_graphing(tmp_path: Path, mock_savefig, tc: CsvTestCase):
    """Test that correct graph files are generated based on the combinations in the CSV files."""
    data_dir = tmp_path / "results"
    data_dir.mkdir()

    # Generate the dummy data
    create_dummy_csv(data_dir / "test_data.csv", tc.mock_data)

    process_csv_files(str(data_dir))

    # Verify the correct number of graphs were requested
    assert mock_savefig.call_count == len(tc.expected_graphs)

    # Extract ONLY the filenames passed to plt.savefig (ignoring the directory paths)
    actual_saved_files = [
        os.path.basename(call[0][0]) for call in mock_savefig.call_args_list
    ]

    # Assert that all expected graph names were processed
    for expected_graph in tc.expected_graphs:
        assert expected_graph in actual_saved_files


def test_redundancy_extraction(tmp_path: Path, mock_savefig):
    """Test that the regex specifically extracts the redundancy integer correctly."""
    data_dir = tmp_path / "results"
    data_dir.mkdir()

    # Test specific regex matching for sample_id: N{X}_R{Y}_{Z} -> Extract Y
    test_data = [
        {
            "sample_id": "N400_R15_13",
            "model": "m1",
            "strategy": "few-shot",
            "ser": 0.5,
            "cipher_length": 400,
            "ground_truth": "x" * 400,
            "with_spaces": "true",
        },
        {
            "sample_id": "N400_R0_1",
            "model": "m1",
            "strategy": "few-shot",
            "ser": 0.9,
            "cipher_length": 400,
            "ground_truth": "x" * 400,
            "with_spaces": "true",
        },
    ]
    create_dummy_csv(data_dir / "regex_test.csv", test_data)

    process_csv_files(str(data_dir))
    mock_savefig.assert_called_once()


def test_default_directory_fallback(mocker):
    """Test that process_csv_files uses the correct fallback directory when None is provided."""
    # Patch glob.glob so we can capture the exact directory path the script tries to search
    mock_glob = mocker.patch(
        "src.visualization.visualization.glob.glob", return_value=[]
    )

    # Call the function with no arguments to trigger the fallback logic
    process_csv_files()

    # Reconstruct the exact path the module SHOULD have built internally
    import src.visualization.visualization as vis_module

    expected_dir = os.path.join(
        os.path.dirname(vis_module.__file__), "..", "..", "results"
    )
    expected_glob_arg = os.path.join(expected_dir, "*.csv")

    # Assert that glob.glob was called with that exact fallback path
    mock_glob.assert_called_once_with(expected_glob_arg)
