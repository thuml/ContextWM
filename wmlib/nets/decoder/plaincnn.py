import torch
import torch.nn as nn

from .base import BaseDecoder
from ..modules import *
from ... import core


class PlainCNNDecoder(BaseDecoder):

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

    def _cnn(self, features):
        channels = {k: self._shapes[k][-1] for k in self.cnn_keys}
        ConvT = nn.ConvTranspose2d
        x = self.get("convin", nn.Linear, features.shape[-1], 32 * self._cnn_depth)(features)
        x = torch.reshape(x, [-1, 32 * self._cnn_depth, 1, 1]).to(memory_format=torch.channels_last)

        for i, kernel in enumerate(self._cnn_kernels):
            depth = 2 ** (len(self._cnn_kernels) - i - 2) * self._cnn_depth
            act, norm = self._act, self._norm
            if i == len(self._cnn_kernels) - 1:
                depth, act, norm = sum(channels.values()), get_act("none"), "none"
            x = self.get(f"conv{i}", ConvT, x.shape[1], depth, kernel, 2)(x)
            x = self.get(f"convnorm{i}", NormLayer, norm, x.shape[-3:])(x)
            x = act(x)

        x = x.reshape(features.shape[:-1] + x.shape[1:])
        means = torch.split(x, list(channels.values()), 2)
        dists = {
            key: core.dists.Independent(core.dists.MSE(mean), 3)
            for (key, shape), mean in zip(channels.items(), means)
        }
        return dists
