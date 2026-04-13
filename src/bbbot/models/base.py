"""Abstract base class for prediction models."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Interface for all prediction models."""

    name: str

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """Train the model on features X and target y."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions (for classifiers)."""
        ...

    def save(self, path: Path) -> None:
        """Serialize model to disk."""
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load a model from disk."""
        import joblib
        return joblib.load(path)
