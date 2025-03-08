from dataclasses import dataclass
import torch
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.utils.net.common import ActorCritic


# Used to eliminate double calculation of the common network, which
# caused to create two gradient tree branches
class CommonNetCache(torch.nn.Module):

    _common_net: torch.nn.Module
    _output_dim: int

    _last_input: torch.Tensor | None
    _last_output: torch.Tensor | None
    _last_output_hidden: torch.Tensor | None

    def __init__(self, common_net: torch.nn.Module, output_dim: int):
        super().__init__()

        self._output_dim = output_dim
        self._common_net = common_net
        self._last_input = None
        self._last_output = None
        self._last_output_hidden = None

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self,
                x: torch.Tensor,
                state=None,
                info={}) -> tuple[torch.Tensor, torch.Tensor | None]:

        if x is self._last_input:
            assert self._last_output is not None
            return self._last_output, self._last_output_hidden

        output, hidden = self._common_net(x, state, info)

        self._last_input = x
        self._last_output = output
        self._last_output_hidden = hidden

        return output, hidden


@dataclass
class DiscreteActorCriticNetworkWithCommonEncoderNetworkParameters:
    action_space_n: int
    actor_hidden_sizes: list[int]
    critic_hidden_sizes: list[int]
    net_output_dim: int


@dataclass
class ActorCriticNetwork:
    actor: torch.nn.Module
    critic: torch.nn.Module
    actor_critic: torch.nn.Module


def create_discrete_actor_critic_net_with_common_encoder_net(
        net: torch.nn.Module, device: torch.device,
        **kwargs) -> ActorCriticNetwork:

    params = DiscreteActorCriticNetworkWithCommonEncoderNetworkParameters(
        **kwargs)

    net = CommonNetCache(net, output_dim=params.net_output_dim)

    actor = Actor(net, [params.action_space_n],
                  hidden_sizes=params.actor_hidden_sizes,
                  preprocess_net_output_dim=params.net_output_dim,
                  device=device).to(device)
    critic = Critic(net,
                                          preprocess_net_output_dim=params.net_output_dim,\
                                          hidden_sizes=params.critic_hidden_sizes,
                                          device=device).to(device)
    actor_critic = ActorCritic(actor, critic)

    return ActorCriticNetwork(actor=actor,
                              critic=critic,
                              actor_critic=actor_critic)
