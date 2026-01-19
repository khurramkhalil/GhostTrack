"""Configuration loader for GhostTrack project."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model configuration."""
    base_model: str = "gpt2"
    d_model: int = 768
    n_layers: int = 12
    device: str = "cuda"


@dataclass
class SAEConfig:
    """Sparse Autoencoder configuration."""
    architecture: str = "JumpReLU"
    d_model: int = 768
    d_hidden: int = 4096
    threshold: float = 0.1
    lambda_sparse: float = 0.01


@dataclass
class SAETrainingConfig:
    """SAE training configuration."""
    epochs: int = 20
    batch_size: int = 2048
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    num_tokens: int = 100_000_000
    max_length: int = 512
    validation_every: int = 1000
    target_recon_loss: float = 0.01
    target_sparsity_min: int = 50
    target_sparsity_max: int = 100


@dataclass
class TrackingConfig:
    """Hypothesis tracking configuration."""
    top_k_features: int = 50
    semantic_weight: float = 0.6
    activation_weight: float = 0.2
    position_weight: float = 0.2
    association_threshold: float = 0.5
    birth_threshold: float = 0.5
    death_threshold: float = 0.1


@dataclass
class DetectionConfig:
    """Hallucination detection configuration."""
    entropy_threshold: float = 1.5
    churn_threshold: float = 0.3
    entropy_weight: float = 0.4
    churn_weight: float = 0.3
    ml_weight: float = 0.3
    threshold: float = 0.5


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    primary: str = "truthful_qa"
    split_train: float = 0.7
    split_val: float = 0.15
    split_test: float = 0.15
    stratify_by: str = "category"


@dataclass
class PathsConfig:
    """Paths configuration."""
    data_dir: str = "./data"
    cache_dir: str = "./data/cache"
    models_dir: str = "./models/checkpoints"
    results_dir: str = "./results"
    logs_dir: str = "./logs"

    def __post_init__(self):
        """Ensure all paths exist."""
        for path_name in ['data_dir', 'cache_dir', 'models_dir', 'results_dir', 'logs_dir']:
            path = Path(getattr(self, path_name))
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    sae_training: SAETrainingConfig = field(default_factory=SAETrainingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            sae=SAEConfig(**config_dict.get('sae', {})),
            sae_training=SAETrainingConfig(**config_dict.get('sae_training', {})),
            tracking=TrackingConfig(**config_dict.get('tracking', {})),
            detection=DetectionConfig(**config_dict.get('detection', {})),
            dataset=DatasetConfig(**config_dict.get('dataset', {})),
            paths=PathsConfig(**config_dict.get('paths', {}))
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            'model': self.model.__dict__,
            'sae': self.sae.__dict__,
            'sae_training': self.sae_training.__dict__,
            'tracking': self.tracking.__dict__,
            'detection': self.detection.__dict__,
            'dataset': self.dataset.__dict__,
            'paths': self.paths.__dict__
        }


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses .claude file.

    Returns:
        Config object with all settings.
    """
    if config_path is None:
        # Default to .claude file in project root
        config_path = Path(__file__).parent.parent / '.claude'

    config_path = Path(config_path)

    if not config_path.exists():
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return Config()

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config.from_dict(config_dict)


def save_config(config: Config, path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Config object to save.
        path: Path where to save the config.
    """
    config_dict = config.to_dict()

    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
