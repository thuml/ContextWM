
import torch
from torch import jit
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
from torchtyping import TensorType, patch_typeguard
from typing import Tuple
from typeguard import typechecked

from ... import core
from ..modules import *
from .base import BaseDynamics, State


class EnsembleRSSM(BaseDynamics):
    r"""
    References:
    - Hafner, Danijar, et al. "Learning latent dynamics for planning from pixels."
    - Hafner, Danijar, et al. "Dream to control: Learning behaviors by latent imagination."
    - Hafner, Danijar, et al. "Mastering atari with discrete world models."

    """

    def __init__(
        self,
        action_free=False,
        fill_action=None,  # 50 in apv
        ensemble=5,
        stoch=30,
        deter=200,
        hidden=200,
        discrete=False,
        act="elu",
        norm="none",
        std_act="softplus",
        min_std=0.1,
    ):
        super().__init__(
            action_free, fill_action, ensemble, stoch, deter, hidden, discrete, act, norm, std_act, min_std,
        )
        self._cell = torch.jit.script(GRUCell(self._hidden, self._deter, norm=True))

    def initial(self, batch_size: int, device) -> State:
        """
        returns initial RSSM state
        """

        if self._discrete:
            state = dict(
                logit=torch.zeros(batch_size, self._stoch, self._discrete),
                stoch=torch.zeros(batch_size, self._stoch, self._discrete),
                deter=self._cell.get_initial_state(batch_size))
        else:
            state = dict(
                mean=torch.zeros(batch_size, self._stoch),
                std=torch.zeros(batch_size, self._stoch),
                stoch=torch.zeros(batch_size, self._stoch),
                deter=self._cell.get_initial_state(batch_size))
        return core.dict_to_device(state, device)

    # @jit.script_method
    def observe(
        self,
        embed: TensorType["batch", "seq", "emb_dim"],
        action: TensorType["batch", "seq", "act_dim"],
        is_first,
        state: State = None
    ):
        # a permute of (batch, sequence) to (sequence, batch)
        swap = lambda x: torch.permute(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0], action.device)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        post, prior = core.sequence_scan(
            self.obs_step,
            state, action, embed, is_first
        )
        post = {k: swap(v) for k, v in post.items()}  # put to (batch, sequence) again
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def obs_step(
        self,
        prev_state: State,
        prev_action: TensorType["batch", "act_dim"],
        embed: TensorType["batch", "emb_dim"],
        is_first: TensorType["batch"],
        sample=True,
    ) -> Tuple[State, State]:
        maskout = lambda x: torch.einsum("b,b...->b...", 1.0 - is_first.to(x.dtype), x)
        prev_state = core.dict_apply(prev_state, maskout)
        prev_action = maskout(prev_action)

        prior = self.img_step(prev_state, prev_action, sample)
        x = torch.cat([prior["deter"], embed], -1)  # embed is encoder conv output
        x = self.get("obs_out", nn.Linear, x.shape[-1], self._hidden)(x)
        x = self.get("obs_out_norm", NormLayer, self._norm, x.shape[-1])(x)
        x = self._act(x)
        stats = self._suff_stats_layer("obs_dist", x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def img_step(
        self,
        prev_state: State,
        prev_action: TensorType["batch", "act_dim"],
        sample=True,
    ) -> State:
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            prev_stoch = torch.reshape(prev_stoch, (*prev_stoch.shape[:-2], self._stoch * self._discrete))
        x = torch.cat([prev_stoch, self.fill_action_with_zero(prev_action)], -1)
        x = self.get("img_in", nn.Linear, x.shape[-1], self._hidden)(x)
        x = self.get("img_in_norm", NormLayer, self._norm, x.shape[-1])(x)
        x = self._act(x)
        deter = prev_state["deter"]
        x, deter = self._cell(x, deter)
        stats = self._suff_stats_ensemble(x)
        index = int(tdist.Uniform(0, self._ensemble).sample().item())
        stats = {k: v[index] for k, v in stats.items()}
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior


class GRUCell(nn.Module):

    def __init__(self, input_size, size, norm=False, act=torch.tanh, update_bias=-1):
        super().__init__()
        self._size = size
        self._act = get_act(act)
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(input_size + size, 3 * size, bias=norm is not None)
        if norm:
            self._norm = nn.LayerNorm(3 * size, eps=1e-3)  # eps equal to tf

    @property
    def state_size(self):
        return self._size

    @torch.jit.export
    def get_initial_state(self, batch_size: int):  # defined by tf.keras.layers.AbstractRNNCell
        return torch.zeros(batch_size, self._size)

    # @jit.script_method
    def forward(self, input, state):
        parts = self._layer(torch.cat([input, state], -1))
        if self._norm is not False:  # check if jit compatible
            parts = self._norm(parts)
        reset, cand, update = torch.chunk(parts, 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)  # it also multiplies the reset by the input
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, output
