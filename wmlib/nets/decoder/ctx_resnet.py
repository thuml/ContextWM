import torch
import torch.nn as nn

from .base import BaseDecoder
from ..modules import *
from ... import core


class ContextualizedResNetDecoder(BaseDecoder):

    # TODO: remame args
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
        ctx_attmask=0.75,
        ctx_attmaskwarmup=-1,
        **dummy_kwargs,
    ):
        super().__init__(shapes, cnn_keys, mlp_keys, mlp_layers)

        self._cnn_depth = cnn_depth

        self._res_layers = res_layers
        self._res_depth = res_depth
        self._res_norm = res_norm

        self._ctx_attmask = ctx_attmask
        self._ctx_attmaskwarmup = None if ctx_attmaskwarmup == -1 else ctx_attmaskwarmup

        self._training_step = 0
        self._current_attmask = None

    def __call__(self, features, shortcuts=None):
        outputs = {}
        if self.cnn_keys:
            outputs.update(self._cnn(features, shortcuts))
        if self.mlp_keys:
            outputs.update(self._mlp(features))
        return outputs

    # TODO: clean up
    def _cnn(self, features, shortcuts=None):
        if self.training:
            self._training_step += 1

        if self._ctx_attmaskwarmup is not None:
            self._current_attmask = (1 - self._ctx_attmask) * \
                (1 - min(1, self._training_step / self._ctx_attmaskwarmup)) + self._ctx_attmask
            if self._training_step % 100 == 0:
                print(f"Current attention mask: {self._current_attmask} {self._training_step}")
        else:
            self._current_attmask = None

        seq_len = features.shape[1]
        channels = {k: self._shapes[k][-1] for k in self.cnn_keys}

        L = self._res_depth
        hw = 64 // 2**(self._res_depth + 1)
        x = self.get("convin", nn.Linear, features.shape[-1], hw * hw * (2**(L - 1)) * self._cnn_depth)(features)
        x = torch.reshape(x, [-1, (2**(L - 1)) * self._cnn_depth, hw, hw]).to(memory_format=torch.channels_last)
        for i in range(L):
            x = self.get(f"unpool{i}", nn.UpsamplingNearest2d, scale_factor=2)(x)
            depth = x.shape[1]

            ctx = shortcuts[x.shape[2]]
            addin = ctx.reshape(features.shape[0], -1, *ctx.shape[-3:])  # [B, K, C, H, W]
            addin = addin.repeat_interleave(x.shape[0] // addin.shape[0], dim=0)  # [BT, K, C, H, W]
            addin = addin.reshape(-1, *addin.shape[-3:])  # [BTK, C, H, W]

            x = self.get(f"res{i}", ResidualStack, x.shape[1], depth // 2,
                         self._res_layers, norm=self._res_norm, dec=True,
                         addin_dim=addin.shape[1],
                         has_addin=(lambda x: x % 2 == 0) if ctx.shape[-1] < 32 else (lambda x: False),
                         cross_att=True,
                         mask=self._ctx_attmask,
                         spatial_dim=x.shape[-2:],
                         )(x, addin, attmask=self._current_attmask)

        depth = sum(channels.values())
        x = self.get(f"convout", nn.ConvTranspose2d, x.shape[1], depth, 3, 2, 1, output_padding=1)(x)

        x = x.reshape(features.shape[:-1] + x.shape[1:])
        means = torch.split(x, list(channels.values()), 2)
        dists = {
            key: core.dists.Independent(core.dists.MSE(mean), 3)
            for (key, shape), mean in zip(channels.items(), means)
        }
        return dists
