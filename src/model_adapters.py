"""
Model adapter layer for the fMRI LLM pipeline.

BaseModelAdapter         – abstract adapter interface
SklearnAdapter           – wraps any sklearn estimator
PyTorchLightningAdapter  – wraps a Lightning module + Trainer
"""

from abc import ABC, abstractmethod

import torch


class BaseModelAdapter(ABC):
    @abstractmethod
    def fit(self, X, y, **kwargs): pass

    @abstractmethod
    def predict(self, X): pass


class SklearnAdapter(BaseModelAdapter):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class PyTorchLightningAdapter(BaseModelAdapter):
    def __init__(self, lightning_module, trainer):
        self.model = lightning_module
        self.trainer = trainer

    def fit(self, train_loader, val_loader=None, **kwargs):
        # Lightning accepts DataLoaders, not raw X/y arrays
        self.trainer.fit(self.model, train_loader, val_loader)

    def predict(self, X):
        """Handle both single-task (Tensor) and multi-task (Tuple) outputs.
        For multi-task models, returns the first output (fMRI regression head).
        """
        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.tensor(X).float())
            pred = output[0] if isinstance(output, tuple) else output
        return pred.numpy()
