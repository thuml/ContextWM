
from abc import ABC, abstractmethod

import torch
import torch.distributions as tdist
from torchtyping import TensorType, patch_typeguard
from typing import Dict, Tuple, Union
from typeguard import typechecked

from ... import core
from ...core import dists
from ..modules import *


State = Dict[str, torch.Tensor]  # FIXME to be more specified


class BaseDynamics(core.Module, ABC):

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
        super().__init__()
        self._action_free = action_free
        self._fill_action = fill_action
        self._ensemble = ensemble
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._discrete = discrete
        self._act = get_act(act)
        self._norm = norm
        self._std_act = std_act
        self._min_std = min_std

    @abstractmethod
    def initial(self, batch_size: int, device) -> State:
        pass

    def fill_action_with_zero(self, action):
        # action: [*B, action]
        B, D = action.shape[:-1], action.shape[-1]
        if self._action_free:
            return torch.zeros([*B, self._fill_action]).to(action.device)
        else:
            if self._fill_action is not None:
                zeros = torch.zeros([*B, self._fill_action - D]).to(action.device)
                return torch.cat([action, zeros], axis=1)
            else:
                # doing nothing
                return action

    @abstractmethod
    def observe(
        self,
        embed: TensorType["batch", "seq", "emb_dim"],
        action: TensorType["batch", "seq", "act_dim"],
        is_first,
        state: State = None
    ):
        pass

    def imagine(
        self,
        action: TensorType["batch", "seq", "act_dim"],
        state: State = None
    ):
        # a permute of (batch, sequence) to (sequence, batch)
        swap = lambda x: torch.permute(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0], action.device)
        assert isinstance(state, dict), state
        action = swap(action)
        prior = core.sequence_scan(self.img_step, state, action)[0]
        prior = {k: swap(v) for k, v in prior.items() if k != "mems"}
        return prior

    def get_feat(self, state):
        """
        gets stoch and deter as tensor
        """

        # FIXME verify shapes of this function
        stoch = state["stoch"]
        if self._discrete:
            stoch = torch.reshape(stoch, (*stoch.shape[:-2], self._stoch * self._discrete))
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state: State):
        """
        gets the stochastic state distribution
        """
        if self._discrete:
            logit = state["logit"]
            logit = logit.float()
            dist = dists.Independent(dists.OneHotDist(logit), 1)
        else:
            mean, std = state["mean"], state["std"]
            mean = mean.float()
            std = std.float()
            dist = dists.Independent(dists.Normal(mean, std), 1)
        return dist

    @abstractmethod
    def obs_step(
        self,
        prev_state: State,
        prev_action: TensorType["batch", "act_dim"],
        embed: TensorType["batch", "emb_dim"],
        is_first: TensorType["batch"],
        sample=True,
    ) -> Tuple[State, State]:
        pass

    @abstractmethod
    def img_step(
        self,
        prev_state: State,
        prev_action: TensorType["batch", "act_dim"],
        sample=True,
    ) -> State:
        pass

    def kl_loss(self, post: State, prior: State, forward: bool, balance: float, free: float, free_avg: bool):
        kld = tdist.kl_divergence
        sg = core.dict_detach
        lhs, rhs = (prior, post) if forward else (post, prior)
        mix = balance if forward else (1 - balance)

        free = torch.tensor(free)
        if balance == 0.5:
            value = kld(self.get_dist(lhs), self.get_dist(rhs))
            loss = torch.maximum(value, free).mean()
        else:
            value_lhs = value = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
            value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
            if free_avg:
                loss_lhs = torch.maximum(value_lhs.mean(), free)
                loss_rhs = torch.maximum(value_rhs.mean(), free)
            else:
                loss_lhs = torch.maximum(value_lhs, free).mean()
                loss_rhs = torch.maximum(value_rhs, free).mean()
            loss = mix * loss_lhs + (1 - mix) * loss_rhs
        return loss, value

    def _suff_stats_ensemble(self, inp: TensorType["batch", "hidden"]):
        bs = list(inp.shape[:-1])
        inp = inp.reshape([-1, inp.shape[-1]])
        stats = []
        for k in range(self._ensemble):
            x = self.get(f"img_out_{k}", nn.Linear, inp.shape[-1], self._hidden)(inp)
            x = self.get(f"img_out_norm_{k}", NormLayer, self._norm, x.shape[-1])(x)
            x = self._act(x)
            stats.append(self._suff_stats_layer(f"img_dist_{k}", x))
        stats = {
            k: torch.stack([x[k] for x in stats], 0)
            for k, v in stats[0].items()
        }
        stats = {
            k: v.reshape([v.shape[0]] + bs + list(v.shape[2:]))
            for k, v in stats.items()
        }
        return stats

    def _suff_stats_layer(self, name, x: TensorType["batch", "hidden"]):
        if self._discrete:
            x = self.get(name, nn.Linear, x.shape[-1], self._stoch * self._discrete)(x)
            logit = torch.reshape(x, (*x.shape[:-1], self._stoch, self._discrete))
            return {"logit": logit}
        else:
            x = self.get(name, nn.Linear, x.shape[-1], 2 * self._stoch)(x)
            mean, std = torch.chunk(x, 2, -1)
            std = {
                "softplus": lambda: F.softplus(std),
                "sigmoid": lambda: F.sigmoid(std),
                "sigmoid2": lambda: 2 * F.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}
