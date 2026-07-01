"""Training utilities for v2 System1/System2 baselines."""

from .examples import (
    SYSTEM1_SYSTEM_PROMPT,
    SYSTEM2_SYSTEM_PROMPT,
    TrainingExample,
    load_training_examples,
    transition_to_example,
)

__all__ = [
    "SYSTEM1_SYSTEM_PROMPT",
    "SYSTEM2_SYSTEM_PROMPT",
    "TrainingExample",
    "load_training_examples",
    "transition_to_example",
]
