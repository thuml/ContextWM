import torch
import torch.nn as nn

from .base import BaseEncoder
from ..modules import *


class ResNetEncoder(BaseEncoder):

    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        act="elu",
        cnn_depth=48,
        mlp_layers=[400, 400, 400, 400],
        res_layers=2,
        res_depth=3,
        res_norm='none',
        **dummy_kwargs,
    ):
        super().__init__(shapes, cnn_keys, mlp_keys, mlp_layers)

        self._act = get_act(act)
        self._cnn_depth = cnn_depth

        self._res_layers = res_layers
        self._res_depth = res_depth
        self._res_norm = res_norm

    def _cnn(self, data):
        x = torch.cat(list(data.values()), -1)
        x = x.to(memory_format=torch.channels_last)

        L = self._res_depth
        x = self.get(f"convin", nn.Conv2d, x.shape[1], self._cnn_depth, 3, 2, 1)(x)
        x = self._act(x)

        for i in range(L):
            depth = 2 ** i * self._cnn_depth
            x = self.get(f"res{i}", ResidualStack, x.shape[1], depth,
                         self._res_layers, norm=self._res_norm)(x)
            x = self.get(f"pool{i}", nn.AvgPool2d, 2, 2)(x)

        return x.reshape(tuple(x.shape[:-3]) + (-1,))
