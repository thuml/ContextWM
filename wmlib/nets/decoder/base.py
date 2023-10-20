import re
from abc import ABC, abstractmethod

import torch.nn as nn

from ..modules import *
from ... import core


class BaseDecoder(core.Module, ABC):

    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        mlp_layers=[400, 400, 400, 400],
    ):
        super().__init__()
        self._shapes = shapes
        self.cnn_keys = [
            k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3
        ]
        self.mlp_keys = [
            k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1
        ]
        print("Decoder CNN outputs:", list(self.cnn_keys))
        print("Decoder MLP outputs:", list(self.mlp_keys))
        
        self._mlp_layers = mlp_layers

    def __call__(self, features):
        outputs = {}
        if self.cnn_keys:
            outputs.update(self._cnn(features))
        if self.mlp_keys:
            outputs.update(self._mlp(features))
        return outputs

    @abstractmethod
    def _cnn(self, features):
        pass

    def _mlp(self, features):
        shapes = {k: self._shapes[k] for k in self.mlp_keys}
        x = features
        for i, width in enumerate(self._mlp_layers):
            x = self.get(f"dense{i}", nn.Linear, x.shape[-1], width)(x)
            x = self.get(f"densenorm{i}", NormLayer, self._norm, x.shape[-1:])(x)
            x = self._act(x)
        dists = {}
        for key, shape in shapes.items():
            dists[key] = self.get(f"dense_{key}", DistLayer, shape)(x)
        return dists
