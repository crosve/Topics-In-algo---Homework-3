"""Base interfaces for homework algorithms."""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class HomeworkAgent(ABC):
    """Minimal agent API used by train/eval scripts."""

    @abstractmethod
    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError("Students should implement model serialization.")

    def load(self, path: str) -> None:
        raise NotImplementedError("Students should implement model deserialization.")
