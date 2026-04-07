import yaml
from pathlib import Path
from dotenv import load_dotenv

# Assuming you put the schema above in src/config_schema.py
from src.config_schema import AppConfig
from src.data_handler import DataHandler
from src.model_client import ModelConfig
from src.litellm_client import LiteLLMClient
from src.openrouter_client import OpenRouterClient
from src.evaluator import Evaluator
from src.experiment_pipeline import ExperimentPipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Load the configuration from the specified file.

    Args:
        config_path (str, optional): The path to the configuration file.
            Defaults to "config.yaml".

    Returns:
        AppConfig: The parsed configuration.

    """
    with open(config_path) as f:
        raw_dict = yaml.safe_load(f)
    return AppConfig.from_yaml_dict(raw_dict)


def main() -> None:
    """Start the application."""
    load_dotenv()
    config = load_config()

    logger.info("Initializing experiment pipeline...")

    handler = DataHandler(
        data_path=Path(config.paths.data),
        few_shot_data_path=Path(config.paths.few_shot_data),
        system_prompt=config.prompts.system_prompt,
    )
    handler.load_data(with_spaces=config.experiment.with_spaces)

    model_cfg = ModelConfig(
        model_name=config.model.name,
        temperature=config.model.temperature,
        max_tokens=config.model.max_tokens,
        api_base=config.model.api_base,
        max_retries=config.model.max_retries,
        timeout=config.model.timeout,
        pacing_delay=config.model.pacing_delay,
        initial_backoff=config.model.initial_backoff,
        backoff_factor=config.model.backoff_factor,
    )
    if config.model.name.startswith("openrouter"):
        model_cfg.model_name = config.model.name.removeprefix("openrouter/")
        client = OpenRouterClient(config=model_cfg)
    else:
        client = LiteLLMClient(config=model_cfg)


    evaluator = Evaluator()

    pipeline = ExperimentPipeline(
        handler=handler,
        client=client,
        evaluator=evaluator,
        output_dir=Path(config.paths.output_dir),
    )

    pipeline.run(
        batch_size=config.experiment.batch_size,
        strategy=config.experiment.strategy,
    )


if __name__ == "__main__":
    main()
