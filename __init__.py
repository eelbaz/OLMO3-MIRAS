"""
OLMo3-MIRAS: Integration of MIRAS Neural Long-Term Memory with OLMo3.
"""

from .model.olmo3_miras import (
    OLMo3MIRASConfig,
    OLMo3MIRASModel,
    OLMo3MIRASForCausalLM
)
from .memory.neural_memory import (
    MIRASMemoryConfig,
    NeuralLongTermMemory,
    PersistentMemory
)

__version__ = "0.1.0"
__all__ = [
    "OLMo3MIRASConfig",
    "OLMo3MIRASModel",
    "OLMo3MIRASForCausalLM",
    "MIRASMemoryConfig",
    "NeuralLongTermMemory",
    "PersistentMemory"
]
