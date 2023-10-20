
import re
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..modules import *
from ... import core


class BaseEncoder(core.Module, ABC):

    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        mlp_layers=[400, 400, 400, 400],
    ):
        super().__init__()
        self.shapes = shapes
        self.cnn_keys = [
            k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3
        ]
        self.mlp_keys = [
            k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1
        ]
        print("Encoder CNN inputs:", list(self.cnn_keys))
        print("Encoder MLP inputs:", list(self.mlp_keys))

        self._mlp_layers = mlp_layers

    def __call__(self, data):
        key, shape = list(self.shapes.items())[0]
        batch_dims = data[key].shape[:-len(shape)]
        data = {
            k: torch.reshape(v, (-1,) + tuple(v.shape)[len(batch_dims):])
            for k, v in data.items()
        }
        outputs = []
        if self.cnn_keys:
            outputs.append(self._cnn({k: data[k] for k in self.cnn_keys}))
        if self.mlp_keys:
            outputs.append(self._mlp({k: data[k] for k in self.mlp_keys}))
        output = torch.cat(outputs, -1)
        return output.reshape(batch_dims + output.shape[1:])

    @abstractmethod
    def _cnn(self, data):
        pass

    def _mlp(self, data):
        x = torch.cat(list(data.values()), -1)
        for i, width in enumerate(self._mlp_layers):
            x = self.get(f"dense{i}", nn.Linear, x.shape[-1], width)(x)
            x = self.get(f"densenorm{i}", NormLayer, self._norm, x.shape[-1:])(x)
            x = self._act(x)
        return x
