from dataclasses import dataclass
from typing import Literal


@dataclass
class PromptsConfig:
    """Configuration for prompts."""

    system_prompt: str


@dataclass
class PathsConfig:
    """Configuration for paths."""

    data: str
    few_shot_data: str
    output_dir: str


@dataclass
class ModelParamsConfig:
    """Configuration for the model."""

    name: str
    temperature: float = 0.0
    max_tokens: int = 4096
    max_retries: int = 3
    timeout: int = 120
    pacing_delay: float = 0.0
    initial_backoff: float = 2.0
    backoff_factor: float = 2.0
    api_base: str | None = None


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""

    batch_size: int
    strategy: Literal["zero-shot", "few-shot"]
    with_spaces: bool


@dataclass
class AppConfig:
    """Configuration for the application."""

    prompts: PromptsConfig
    paths: PathsConfig
    model: ModelParamsConfig
    experiment: ExperimentConfig

    @classmethod
    def from_yaml_dict(cls, data: dict) -> "AppConfig":
        """Instantiate the nested dataclasses from the raw YAML dictionary."""
        return cls(
            prompts=PromptsConfig(**data.get("prompts", {})),
            paths=PathsConfig(**data.get("paths", {})),
            model=ModelParamsConfig(**data.get("model", {})),
            experiment=ExperimentConfig(**data.get("experiment", {})),
        )
