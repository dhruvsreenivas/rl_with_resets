"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from continuous_control.agents.sac import temperature
from continuous_control.agents.sac.actor import update as update_actor
from continuous_control.agents.sac.critic import target_update
from continuous_control.agents.sac.critic import update as update_critic
from continuous_control.datasets import Batch
from continuous_control.networks import critic_net, policies
from continuous_control.networks.common import InfoDict, Model, PRNGKey


@functools.partial(jax.jit, static_argnames=("update_target", "use_mean_reduction"))
def _update_jit(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temp: Model,
    batch: Batch,
    discount: float,
    tau: float,
    target_entropy: float,
    update_target: bool,
    use_mean_reduction: bool,
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(
        key,
        actor,
        critic,
        target_critic,
        temp,
        batch,
        discount,
        soft_critic=True,
        use_mean_reduction=use_mean_reduction,
    )
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)
    new_temp, alpha_info = temperature.update(
        temp, actor_info["entropy"], target_entropy
    )

    return (
        rng,
        new_actor,
        new_critic,
        new_target_critic,
        new_temp,
        {**critic_info, **actor_info, **alpha_info},
    )


class SACLearner(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        target_update_period: int = 1,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        use_mean_reduction: bool = False,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
        actor_def = policies.NormalTanhPolicy(hidden_dims, action_dim)
        actor = Model.create(
            actor_def,
            inputs=[actor_key, observations],
            tx=optax.adam(learning_rate=actor_lr),
        )

        critic_def = critic_net.DoubleCritic(hidden_dims)
        critic = Model.create(
            critic_def,
            inputs=[critic_key, observations, actions],
            tx=optax.adam(learning_rate=critic_lr),
        )
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions]
        )

        temp = Model.create(
            temperature.Temperature(init_temperature),
            inputs=[temp_key],
            tx=optax.adam(learning_rate=temp_lr),
        )

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.rng = rng
        self.use_mean_reduction = use_mean_reduction

        self.step = 1

    def sample_actions(
        self, observations: np.ndarray, temperature: float = 1.0
    ) -> jnp.ndarray:
        rng, actions = policies.sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, observations, temperature
        )
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        self.step += 1

        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit(
            self.rng,
            self.actor,
            self.critic,
            self.target_critic,
            self.temp,
            batch,
            self.discount,
            self.tau,
            self.target_entropy,
            self.step % self.target_update_period == 0,
            self.use_mean_reduction,
        )

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        return info
