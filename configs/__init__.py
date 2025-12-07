"""
OLMo3-MIRAS Model Configurations.

Provides pre-configured model architectures following OLMo3 design patterns
with MIRAS neural long-term memory integration for unlimited context.
"""

from .model_configs import (
    OLMO3_MIRAS_500M_CONFIG,
    OLMO3_MIRAS_1B_CONFIG,
    get_model_config,
    get_training_config,
    TrainingStage,
)

__all__ = [
    "OLMO3_MIRAS_500M_CONFIG",
    "OLMO3_MIRAS_1B_CONFIG",
    "get_model_config",
    "get_training_config",
    "TrainingStage",
]
