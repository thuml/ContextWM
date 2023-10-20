import torch
import torch.nn as nn

from .base import BaseEncoder
from ..modules import *


class PlainCNNEncoder(BaseEncoder):

    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        act="elu",
        norm="none",
        cnn_depth=48,
        cnn_kernels=(4, 4, 4, 4),
        mlp_layers=[400, 400, 400, 400],
        **dummy_kwargs,
    ):
        super().__init__(shapes, cnn_keys, mlp_keys, mlp_layers)

        self._act = get_act(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels

    def _cnn(self, data):
        x = torch.cat(list(data.values()), -1)
        x = x.to(memory_format=torch.channels_last)
        for i, kernel in enumerate(self._cnn_kernels):
            depth = 2 ** i * self._cnn_depth
            x = self.get(f"conv{i}", nn.Conv2d, x.shape[1], depth, kernel, 2)(x)
            x = self.get(f"convnorm{i}", NormLayer, self._norm, x.shape[-3:])(x)
            x = self._act(x)
        return x.reshape(tuple(x.shape[:-3]) + (-1,))
