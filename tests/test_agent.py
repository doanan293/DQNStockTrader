import numpy as np
import pytest
import torch
from torch import nn
from typing_extensions import override

from dqn_stock_trader.agent import AgentConfig, DQNAgent, Variant
from dqn_stock_trader.network import QNetwork
from dqn_stock_trader.replay import ReplayBuffer, Transition

STATE_SIZE = 11
N_ACTIONS = 5


def _transition(seed: int) -> Transition:
    rng = np.random.default_rng(seed)
    return Transition(
        state=rng.random(STATE_SIZE),
        action=int(rng.integers(N_ACTIONS)),
        reward=float(rng.normal()),
        next_state=rng.random(STATE_SIZE),
        done=False,
    )


def _agent(
    variant: Variant = "double", gamma: float = 0.95, target_sync_steps: int = 1_000
) -> DQNAgent:
    config = AgentConfig(
        variant=variant, gamma=gamma, batch_size=4, target_sync_steps=target_sync_steps
    )
    agent = DQNAgent(STATE_SIZE, N_ACTIONS, config, seed=0)
    for i in range(4):
        agent.remember(_transition(i))
    return agent


def test_network_output_shape() -> None:
    net = QNetwork(STATE_SIZE, N_ACTIONS)
    assert net(torch.zeros(4, STATE_SIZE)).shape == (4, N_ACTIONS)


def test_buffer_capacity_and_sample() -> None:
    buffer = ReplayBuffer(capacity=3, seed=0)
    for i in range(5):
        buffer.push(_transition(i))
    assert len(buffer) == 3
    batch = buffer.sample(2)
    assert batch.states.shape == (2, STATE_SIZE)
    assert batch.actions.shape == (2,)


def _params_equal(a: nn.Module, b: nn.Module) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters(), strict=True))


def test_target_syncs_every_n_steps() -> None:
    agent = _agent(variant="fixed_target", target_sync_steps=2)
    agent.learn()
    assert not _params_equal(agent.online, agent.target)
    agent.learn()
    assert _params_equal(agent.online, agent.target)


def test_dqn_variant_uses_online_net_as_target() -> None:
    agent = _agent(variant="dqn")
    assert agent.target is agent.online


def test_learn_returns_none_until_buffer_has_a_batch() -> None:
    agent = DQNAgent(STATE_SIZE, N_ACTIONS, AgentConfig(batch_size=4), seed=0)
    agent.remember(_transition(0))
    assert agent.learn() is None


class _Fixed(nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.values = torch.tensor(values)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.values.expand(x.shape[0], -1)


def test_double_dqn_selects_action_with_online_net() -> None:
    agent = _agent(variant="double", gamma=0.5)
    agent.online = _Fixed([0.0, 9.0, 0.0, 0.0, 0.0])
    agent.target = _Fixed([7.0, 2.0, 0.0, 0.0, 0.0])
    batch = agent.buffer.sample(4)
    targets = agent.compute_targets(batch)
    assert torch.allclose(targets, batch.rewards + 0.5 * 2.0)


def test_greedy_act_is_deterministic() -> None:
    agent = _agent()
    state = np.linspace(0, 1, STATE_SIZE)
    actions = {agent.act(state, greedy=True) for _ in range(10)}
    assert len(actions) == 1


def test_epsilon_decays_after_learning() -> None:
    agent = _agent()
    agent.learn()
    assert agent.epsilon == pytest.approx(0.995)
