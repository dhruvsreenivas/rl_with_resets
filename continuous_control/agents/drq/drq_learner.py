"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from continuous_control.agents.drq.augmentations import batched_random_crop
from continuous_control.agents.drq.networks import DrQDoubleCritic, DrQPolicy
from continuous_control.agents.sac import temperature
from continuous_control.agents.sac.actor import update as update_actor
from continuous_control.agents.sac.actor import update_bc as update_actor_bc
from continuous_control.agents.sac.critic import target_update
from continuous_control.agents.sac.critic import update as update_critic
from continuous_control.datasets import Batch
from continuous_control.networks import policies
from continuous_control.networks.common import (
    InfoDict,
    Model,
    ModelDecoupleOpt,
    PRNGKey,
)


@functools.partial(
    jax.jit,
    static_argnames=(
        "update_target",
        "decoupled",
        "aug_actor_observations",
        "aug_critic_observations",
        "use_mean_reduction",
        "use_max_reduction",
    ),
)
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
    decoupled: bool,
    aug_actor_observations: bool,
    aug_critic_observations: bool,
    use_mean_reduction: bool,
    use_max_reduction: bool,
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    aug_observations = batched_random_crop(key, batch.observations)
    rng, key = jax.random.split(rng)
    aug_next_observations = batched_random_crop(key, batch.next_observations)

    # create actor and critic batches
    if aug_actor_observations:
        actor_batch = batch._replace(
            observations=aug_observations, next_observations=aug_next_observations
        )
    else:
        actor_batch = batch

    if aug_critic_observations:
        critic_batch = batch._replace(
            observations=aug_observations, next_observations=aug_next_observations
        )
    else:
        critic_batch = batch

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(
        key,
        actor,
        critic,
        target_critic,
        temp,
        critic_batch,
        discount,
        soft_critic=True,
        use_mean_reduction=use_mean_reduction,
        use_max_reduction=use_max_reduction,
    )
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    # Use critic conv layers in actor if we're training in a coupled fashion and if the shared encoder exists
    if not decoupled and hasattr(new_critic.params, "SharedEncoder"):
        # this should work with regular dicts.
        new_actor_params = flax.core.copy(
            actor.params,
            add_or_replace={"SharedEncoder": new_critic.params["SharedEncoder"]},
        )
        actor = actor.replace(params=new_actor_params)

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, actor_batch)
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


@functools.partial(jax.jit, static_argnames=("aug_actor_observations"))
def _update_jit_bc(
    rng: PRNGKey,
    actor: Model,
    batch: Batch,
    aug_actor_observations: bool,
) -> Tuple[PRNGKey, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    aug_observations = batched_random_crop(key, batch.observations)
    rng, key = jax.random.split(rng)
    aug_next_observations = batched_random_crop(key, batch.next_observations)

    if aug_actor_observations:
        batch = batch._replace(
            observations=aug_observations, next_observations=aug_next_observations
        )

    rng, update_key = jax.random.split(rng)
    new_actor, actor_info = update_actor_bc(update_key, actor, batch)
    return rng, new_actor, actor_info


class DrQLearner(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        actor_hidden_dims: Sequence[int] = (256, 256),
        critic_cnn_features: Sequence[int] = (32, 32, 32, 32),
        critic_cnn_strides: Sequence[int] = (2, 1, 1, 1),
        actor_cnn_features: Optional[Sequence[int]] = None,
        actor_cnn_strides: Optional[Sequence[int]] = None,
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        discount: float = 0.99,
        tau: float = 0.005,
        target_update_period: int = 1,
        target_entropy: Optional[float] = None,
        init_temperature: float = 0.1,
        decoupled: bool = False,
        pass_grads_through_actor_encoder: bool = True,
        pass_grads_through_critic_encoder: bool = True,
        critic_use_layer_norm: bool = False,
        aug_actor_observations: bool = True,
        aug_critic_observations: bool = True,
        use_mean_reduction: bool = False,
        use_max_reduction: bool = False,
    ):

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        # assert that if we're in the coupled case, the actor and critic features/strides match up
        if not decoupled:
            assert actor_cnn_features is None or (
                actor_cnn_features is not None
                and actor_cnn_features == critic_cnn_features
            )
            assert actor_cnn_strides is None or (
                actor_cnn_features is not None
                and actor_cnn_strides == critic_cnn_strides
            )

        # set actor features/strides appropriately: if they are None, they default to critic features/strides, so same encoder architecture is used
        if actor_cnn_features is None:
            actor_cnn_features = critic_cnn_features
        if actor_cnn_strides is None:
            actor_cnn_strides = critic_cnn_strides

        actor_def = DrQPolicy(
            actor_hidden_dims,
            action_dim,
            actor_cnn_features,
            actor_cnn_strides,
            cnn_padding,
            latent_dim,
            pass_grads_through_encoder=decoupled and pass_grads_through_actor_encoder,
        )
        if not decoupled:
            actor = Model.create(
                actor_def,
                inputs=[actor_key, observations],
                tx=optax.adam(learning_rate=actor_lr),
            )
        else:
            actor = ModelDecoupleOpt.create(
                actor_def,
                inputs=[actor_key, observations],
                tx=optax.adam(learning_rate=actor_lr),
                tx_enc=optax.adam(learning_rate=actor_lr),
            )

        critic_def = DrQDoubleCritic(
            critic_hidden_dims,
            critic_cnn_features,
            critic_cnn_strides,
            cnn_padding,
            latent_dim,
            use_layer_norm=critic_use_layer_norm,
            pass_grads_through_encoder=pass_grads_through_critic_encoder,
        )

        critic = ModelDecoupleOpt.create(
            critic_def,
            inputs=[critic_key, observations, actions],
            tx=optax.adam(learning_rate=critic_lr),
            tx_enc=optax.adam(learning_rate=critic_lr),
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
        self.decoupled = decoupled
        self.aug_actor_observations = aug_actor_observations
        self.aug_critic_observations = aug_critic_observations
        self.use_mean_reduction = use_mean_reduction
        self.use_max_reduction = use_max_reduction
        self.step = 0

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
            self.decoupled,
            self.aug_actor_observations,
            self.aug_critic_observations,
            self.use_mean_reduction,
            self.use_max_reduction,
        )

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        return info

    def update_bc(self, batch: Batch) -> InfoDict:
        self.step += 1

        new_rng, new_actor, info = _update_jit_bc(
            self.rng, self.actor, batch, self.aug_actor_observations
        )

        self.rng = new_rng
        self.actor = new_actor

        return info
