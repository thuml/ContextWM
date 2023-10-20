import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import core
from ..core import dists


def Normalize(in_channels: int, norm: str = 'batch', spatial_dim = None) -> nn.Module:
    if norm == 'batch':
        return nn.BatchNorm2d(num_features=in_channels)
    elif norm == 'layer':
        return nn.LayerNorm(normalized_shape=[in_channels, *spatial_dim])
    elif norm == 'group':
        return nn.GroupNorm(num_groups=8, num_channels=in_channels)
    elif norm == 'none':
        return nn.Identity()
    else:
        raise NotImplementedError


class ResidualLayer(core.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, out_dim, norm='batch', addin_dim=0, cross_att=False, mask=0.75, dec=False, spatial_dim=None):
        super(ResidualLayer, self).__init__()

        self.cross_att = cross_att
        self.mask = mask
        self.dec = dec
        if addin_dim != 0 and cross_att:
            self.cross_attention = CrossAttention(in_dim if dec else out_dim)
            self.norm_cross_att = Normalize(in_dim if dec else out_dim, norm=norm, spatial_dim=spatial_dim)
            addin_dim = 0

        if in_dim != out_dim:
            self.identity = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, bias=False),
                Normalize(out_dim, norm=norm, spatial_dim=spatial_dim),
            )
        else:
            self.identity = nn.Identity()

        if norm != 'none':
            self.res_block = nn.Sequential(
                nn.Conv2d(in_dim + addin_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
                Normalize(out_dim, norm=norm, spatial_dim=spatial_dim),
                nn.ReLU(),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
                Normalize(out_dim, norm=norm, spatial_dim=spatial_dim),
            )
        else:
            self.res_block = nn.Sequential(
                nn.Conv2d(in_dim + addin_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            )

        self.out_dim = out_dim

    def forward(self, x, addin=None, temp_align=None, attmask=None):
        if self.cross_att:
            mask = self.mask if attmask is None else attmask
            z = x

            if addin is not None and int(addin.shape[-1] * addin.shape[-2] * (1 - mask)) > 0:
                # For decoder
                # assert(z.shape[1] == addin.shape[1])

                # kv = addin.permute(0, 2, 3, 1).reshape(addin.shape[0], -1, addin.shape[1])  # [B, HW, C]
                # [B*K, C, H, W] => [B*K, H, W, C] => [B, K, HW, C] (commonly K=1)
                kv = addin.permute(0, 2, 3, 1).reshape(z.shape[0], -1, addin.shape[2] * addin.shape[3], addin.shape[1])
                if 'kv_pos_emb' not in self._parameters:
                    self._parameters['kv_pos_emb'] = nn.parameter.Parameter(
                        torch.zeros(kv.shape[-2:]).to(kv.device), requires_grad=True)  # [HW, C]
                kv = kv + self.kv_pos_emb
                kv = kv.reshape(z.shape[0], -1, addin.shape[1])  # [B, KHW, C] (commonly K=1)
                kv = random_mask(kv, mask, temp_align)
                q = z.permute(0, 2, 3, 1).reshape(z.shape[0], -1, z.shape[1])  # [B, HW, C]
                if 'q_pos_emb' not in self._parameters:
                    self._parameters['q_pos_emb'] = nn.parameter.Parameter(
                        torch.zeros(q.shape[-2:]).to(q.device), requires_grad=True)  # [HW, C]
                q = q + self.q_pos_emb
                attn_out, attn_weight = self.cross_attention(q, kv)
                attn_out = attn_out.permute(0, 2, 1).reshape(z.shape)
                attn_out = self.norm_cross_att(attn_out)
                z = torch.relu(z + attn_out)

            x = self.identity(x) + self.res_block(z)
        else:
            if addin is not None:
                x = self.identity(x) + self.res_block(torch.cat([x, addin], dim=1))
            else:
                x = self.identity(x) + self.res_block(x)
        return F.relu(x)


class ResidualStack(core.Module):

    def __init__(self, in_dim, out_dim, n_res_layers=1, norm='batch', dec=False, addin_dim=0, has_addin=lambda x: False, cross_att=False, mask=0.75, spatial_dim=None):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.has_addin = has_addin
        if dec:
            self.stack = nn.ModuleList([
                ResidualLayer(
                    in_dim,
                    out_dim if i == n_res_layers - 1 else in_dim,
                    norm=norm,
                    addin_dim=addin_dim if has_addin(i) else 0,
                    cross_att=cross_att,
                    dec=dec,
                    mask=mask,
                    spatial_dim=spatial_dim,
                ) for i in range(n_res_layers)
            ])
        else:
            self.stack = nn.ModuleList([
                ResidualLayer(
                    in_dim if i == 0 else out_dim,
                    out_dim,
                    norm=norm,
                    addin_dim=addin_dim if has_addin(i) else 0,
                    cross_att=cross_att,
                    dec=dec,
                    mask=mask,
                    spatial_dim=spatial_dim,
                ) for i in range(n_res_layers)
            ])

    def forward(self, x, addin=None, temp_align=None, attmask=None):
        for i, layer in enumerate(self.stack):
            x = layer(x, addin=addin if self.has_addin(i) else None, temp_align=temp_align, attmask=attmask)
        return x


class CrossAttention(core.Module):
    def __init__(self, hidden_size, num_head=4, dropout=0.1, scale=True):
        super().__init__()

        self.att = nn.MultiheadAttention(hidden_size, num_head, dropout=dropout, batch_first=True)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query,
        key,
        output_attentions=False,
    ):
        attn_output, attn_outputs = self.att(query, key, key)
        a = self.resid_dropout(attn_output)
        outputs = [a] + [attn_outputs, ]
        return outputs  # a, (attentions)


def random_mask(x, mask_rate=0.0, seq_len=None):
    if mask_rate == 0:
        return x
    # TODO: accelerate?
    # x: [B, N, C]
    # y: [B, S, C]
    B, N, C = x.shape
    S = int(N * (1 - mask_rate))
    if seq_len is not None:
        assert (B % seq_len == 0)
        rand = torch.rand(B // seq_len, N).to(x.device)
        rand = rand.repeat_interleave(seq_len, dim=0)
    else:
        rand = torch.rand(B, N).to(x.device)
    batch_rand_perm = rand.argsort(dim=1)
    index = batch_rand_perm[:, :S, None].expand(-1, -1, C)
    y = torch.gather(x, 1, index)
    return y


class MLP(core.Module):

    def __init__(self, shape, layers, units, act="elu", norm="none", **out):
        super().__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        self._layers = layers
        self._units = units
        self._norm = norm
        self._act = get_act(act)
        self._out = out

    def __call__(self, features):
        x = features
        x = x.reshape([-1, x.shape[-1]])
        for index in range(self._layers):
            x = self.get(f"dense{index}", nn.Linear, x.shape[-1], self._units)(x)
            x = self.get(f"norm{index}", NormLayer, self._norm, x.shape[-1:])(x)
            x = self._act(x)
        x = x.reshape([*features.shape[:-1], x.shape[-1]])
        return self.get("out", DistLayer, self._shape, **self._out)(x)


class DistLayer(core.Module):

    def __init__(self, shape, dist="mse", min_std=0.1, init_std=0.0):
        super(DistLayer, self).__init__()
        self._shape = shape  # shape can be [], its equivalent to 1.0 in np.prod
        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std

    def __call__(self, inputs):
        out = self.get("out", nn.Linear, inputs.shape[-1], int(np.prod(self._shape)))(inputs)
        out = out.reshape([*inputs.shape[:-1], *self._shape])
        if self._dist in ("normal", "tanh_normal", "trunc_normal"):
            std = self.get("std", nn.Linear, inputs.shape[-1], int(np.prod(self._shape)))(inputs)
            std = std.reshape([*inputs.shape[:-1], *self._shape])
        if self._dist == "mse":
            dist = dists.MSE(out)
            return dists.Independent(dist, len(self._shape))
        if self._dist == "normal":
            # NOTE Doesn't make sense std to be negative
            # NOT USED in algorithm
            raise NotImplementedError(self._dist)

            # dist = dists.Normal(out, std) # FIXME std can only be positive
            # return dists.Independent(dist, len(self._shape))
        if self._dist == "binary":
            # NOTE log_prob means binary_cross_entropy_with_logits
            dist = dists.Bernoulli(logits=out, validate_args=False)  # FIXME: validate_args=None? => Error
            return dists.Independent(dist, len(self._shape))
        if self._dist == "tanh_normal":
            # FIXME NOT USED in algorithm
            raise NotImplementedError(self._dist)

            # mean = 5 * torch.tanh(out / 5)
            # std = F.softplus(std + self._init_std) + self._min_std
            # dist = tdist.Normal(mean, std)
            # dist = tdist.TransformedDistribution(dist, common.TanhBijector()) #tfd.TransformedDistribution(dist, common.TanhBijector())
            # dist = tdist.Independent(dist, len(self._shape))
            # return common.SampleDist(dist)
        if self._dist == "trunc_normal":
            std = 2 * torch.sigmoid((std + self._init_std) / 2) + self._min_std
            dist = dists.TruncNormalDist(torch.tanh(out), std, -1, 1)
            return dists.Independent(dist, 1)
        if self._dist == "onehot":
            dist = dists.OneHotDist(logits=out)
            return dist
        raise NotImplementedError(self._dist)


class NormLayer(core.Module):

    def __init__(self, name, normalized_shape):
        super().__init__()
        if name == "none":
            self._layer = None
        elif name == "layer":
            self._layer = nn.LayerNorm(normalized_shape, eps=1e-3)  # eps equal to tf
        else:
            raise NotImplementedError(name)

    def __call__(self, features):
        if not self._layer:
            return features
        return self._layer(features)


def get_act(act):
    if isinstance(act, str):
        name = act
        if name == "none":
            return lambda x: x
        if name == "mish":
            return lambda x: x * torch.tanh(F.softplus(x))
        elif hasattr(F, name):
            return getattr(F, name)
        elif hasattr(torch, name):
            return getattr(torch, name)
        else:
            raise NotImplementedError(name)
    else:
        return act
