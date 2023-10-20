import torch
import torch.nn as nn

from .base import BaseDecoder
from ..modules import *
from ... import core


class ResNetDecoder(BaseDecoder):

    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        cnn_depth=48,
        mlp_layers=[400, 400, 400, 400],
        res_layers=2,
        res_depth=3,
        res_norm='none',
        **dummy_kwargs,
    ):
        super().__init__(shapes, cnn_keys, mlp_keys, mlp_layers)

        self._cnn_depth = cnn_depth
        self._res_layers = res_layers
        self._res_depth = res_depth
        self._res_norm = res_norm

    def _cnn(self, features):

        channels = {k: self._shapes[k][-1] for k in self.cnn_keys}

        L = self._res_depth
        hw = 64 // 2**(self._res_depth + 1)
        x = self.get("convin", nn.Linear, features.shape[-1], hw * hw * (2**(L - 1)) * self._cnn_depth)(features)
        x = torch.reshape(x, [-1, (2**(L - 1)) * self._cnn_depth, hw, hw]).to(memory_format=torch.channels_last)
        for i in range(L):
            x = self.get(f"unpool{i}", nn.UpsamplingNearest2d, scale_factor=2)(x)
            depth = x.shape[1]
            x = self.get(f"res{i}", ResidualStack, depth, depth // 2,
                         self._res_layers, norm=self._res_norm, dec=True)(x)

        depth = sum(channels.values())
        x = self.get(f"convout", nn.ConvTranspose2d, x.shape[1], depth, 3, 2, 1, output_padding=1)(x)

        x = x.reshape(features.shape[:-1] + x.shape[1:])
        means = torch.split(x, list(channels.values()), 2)
        dists = {
            key: core.dists.Independent(core.dists.MSE(mean), 3)
            for (key, shape), mean in zip(channels.items(), means)
        }
        return dists
