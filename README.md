
# Cipher Decryption Tester

This project is a framework for testing the ability of large language models (LLMs) to decipher homophony-based substitution ciphers. It supports both "zero-shot" and "few-shot" learning strategies, allowing for a comprehensive evaluation of model performance.

## Features

- **Zero-shot and Few-shot Learning:** Test models with or without examples.
- **Configurable Models:** Easily switch between different LLMs (defaults to a Gemini model).
- **Comprehensive Evaluation:** Calculates Symbol Error Rate (SER) and exact match accuracy.
- **Data Handling:** Loads data from JSONL files and saves results to CSV.
- **Resumable Experiments:** Can resume an experiment from where it left off.
- **Batch Processing:** Processes multiple samples in batches for efficiency.

## Getting Started

### Prerequisites

- Python 3.12+
- `uv` package manager (recommended)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/zero-few-shot-attempts.git
   cd zero-few-shot-attempts
   ```

2. **Create a virtual environment and install dependencies:**

   Using `uv`:

   ```bash
   uv sync
   ```

   Using `venv` and `pip`:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Set up your API key:**

   Create a `.env` file in the root of the project and add your API key:

   ```
   PROVIDER_API_KEY=your-api-key
   ```

   **Note:** Replace `your-api-key` with your actual API key and `PROVIDER` with the name of the provider you're using (e.g., `OPENAI`, `OPENROUTER`, `ANTHROPIC`, `GEMINI`) . Refer to [this page](https://docs.litellm.ai/docs/providers) for more information about providers and models.

### Usage

1. **Configure your experiment:**

   Edit the `config.yaml` file to set up your experiment. You can configure:
    - The system prompt
    - Paths to your data and output directories
    - Model parameters (name, temperature, etc.)
    - Experiment settings (batch size, strategy, etc.)

2. **Run the experiment:**

   Using `uv`:
   ```bash
   uv run main.py
   ```

   or using `python`:

   ```bash
   python main.py
   ```

   The results will be saved in the directory specified in `config.yaml` (default is `data/results/`).

## Project Structure

```
.
├── config.yaml
├── main.py
├── pyproject.toml
├── README.md
├── src
│   ├── config_schema.py
│   ├── data_handler.py
│   ├── evaluator.py
│   ├── experiment_pipeline.py
│   └── model_client.py
└── test
```

- **`main.py`**: The main entry point for running experiments.
- **`config.yaml`**: The main configuration file.
- **`src/`**: Contains the core application logic.
  - **`experiment_pipeline.py`**: Orchestrates the experiment from data loading to evaluation.
  - **`data_handler.py`**: Loads and prepares the dataset.
  - **`model_client.py`**: Interacts with the LLM.
  - **`evaluator.py`**: Evaluates the model's predictions.
  - **`config_schema.py`**: Defines the structure of the configuration file.
- **`test/`**: Contains unit tests for the project.

## Configuration

The `config.yaml` file is divided into four main sections:

- **`prompts`**: Contains the system prompt for the LLM.
- **`paths`**: Specifies the paths for the dataset, few-shot examples, and output directory.
- **`model`**: Defines the LLM to use and its parameters.
- **`experiment`**: Controls the experiment's parameters, such as batch size and learning strategy.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
