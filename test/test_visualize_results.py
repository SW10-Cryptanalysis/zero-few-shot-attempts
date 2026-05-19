import matplotlib
matplotlib.use("Agg")
import csv
import pytest
from pathlib import Path

from src.visualize_results import extract_redundancy, parse_csv_files, generate_model_plots, main

@pytest.mark.parametrize(
    "sample_id, expected",
    [
        ("N10000_R25_09", 25),
        ("N5000_R0_12", 0),
        ("N100_R100_1", 100),
        ("invalid_format", None),
        ("", None),
        (None, None),
    ],
)

def test_extract_redundancy(sample_id, expected):
    """Verifies that integers following '_R' are cleanly parsed from various strings."""
    assert extract_redundancy(sample_id) == expected

def test_parse_csv_files(tmp_path):
    """Validates CSV parsing, row skip logic, and dynamic dictionary structures."""
    csv_file = tmp_path / "results_1.csv"

    headers = ["model", "strategy", "sample_id", "ser", "cipher_length", "unrelated_col"]
    rows = [
        ["llama3", "zero-shot", "N1000_R20_01", "0.15", "150", "foo"],
        ["llama3", "few-shot", "N1000_R20_02", "0.05", "150", "bar"],
        ["llama3", "zero-shot", "N1000_R20_03", "0.12", "", "baz"],
        ["gpt4", "zero-shot", "invalid_id", "0.10", "200", "qux"],
        ["gpt4", "zero-shot", "N100_R10_01", "not-a-float", "200", "abc"],
    ]

    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    data = parse_csv_files([str(csv_file)])

    assert "llama3" in data
    assert "gpt4" not in data
    assert "zero-shot" in data["llama3"]
    assert "few-shot" in data["llama3"]

    assert data["llama3"]["zero-shot"]["lengths"] == [150]
    assert data["llama3"]["zero-shot"]["redundancies"] == [20]
    assert data["llama3"]["zero-shot"]["sers"] == [0.15]

def test_generate_model_plots(tmp_path):
    """Ensures Matplotlib successfully draws, saves, and names files correctly."""
    mock_strategies_data = {
        "zero-shot": {"lengths": [100, 200], "redundancies": [10, 20], "sers": [0.4, 0.2]},
        "few-shot": {"lengths": [100, 200], "redundancies": [10, 20], "sers": [0.2, 0.1]}
    }

    model_name = "Llama/3: Test"

    generate_model_plots(model_name, mock_strategies_data, tmp_path)

    expected_filename = "llama_3__test.png"
    expected_path = tmp_path / expected_filename

    assert expected_path.exists()
    assert expected_path.stat().st_size > 0

def test_main_directory_not_found(caplog, mocker):
    """Covers the parser, missing directory validation block, and exit branch."""
    test_args = ["visualize_results.py", "--results_dir", "/this/path/does/not/exist"]
    mocker.patch("sys.argv", test_args)

    with caplog.at_level("INFO"):
        main()

    assert "Error: Directory not found at" in caplog.text


def test_main_no_csv_files(tmp_path, caplog, mocker):
    """Covers the empty directory validation block and exit branch safely."""
    test_args = ["visualize_results.py", "--results_dir", str(tmp_path)]
    mocker.patch("sys.argv", test_args)

    original_path = Path
    def mock_path(value, *args, **kwargs):
        if value == "graphs":
            return original_path(tmp_path)
        return original_path(value, *args, **kwargs)

    mocker.patch("src.visualize_results.Path", side_effect=mock_path)

    with caplog.at_level("INFO"):
        main()

    assert f"No CSV files found in {tmp_path}" in caplog.text


def test_main_full_execution_flow(tmp_path, caplog, mocker):
    """Covers parse_csv_files branch, loops, and successful final execution blocks."""
    valid_csv = tmp_path / "valid_results.csv"
    headers = ["model", "strategy", "sample_id", "ser", "cipher_length"]
    rows = [["TestModel", "zero-shot", "N100_R5_01", "0.2", "50"]]
    with open(valid_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    test_args = ["visualize_results.py", "--results_dir", str(tmp_path)]
    mocker.patch("sys.argv", test_args)

    original_path = Path
    def mock_path(value, *args, **kwargs):
        if value == "graphs":
            return original_path(tmp_path)
        return original_path(value, *args, **kwargs)

    mocker.patch("src.visualize_results.Path", side_effect=mock_path)
    mock_plots = mocker.patch("src.visualize_results.generate_model_plots", autospec=True)

    with caplog.at_level("INFO"):
        main()

    assert mock_plots.called
