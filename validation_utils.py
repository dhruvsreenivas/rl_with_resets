import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from continuous_control.datasets import Batch
from continuous_control.networks.common import Model, PRNGKey


def transform_to_grayscale(batch: Batch) -> Batch:

    def rgb_to_grayscale(arr):
        arr_gs = np.dot(arr, [[0.299], [0.587], [0.114]])
        return arr_gs

    return Batch(
        observations=rgb_to_grayscale(batch.observations),
        actions=batch.actions,
        rewards=batch.rewards,
        next_observations=rgb_to_grayscale(batch.next_observations),
        masks=batch.masks,
    )


@functools.partial(
    jax.jit,
    static_argnames=("use_mean_reduction", "use_max_reduction", "convert_to_grayscale"),
)
def get_validation_metrics(
    actor: Model,
    critic: Model,
    target_critic: Model,
    temp: Model,
    train_batch: Batch,
    val_batch: Batch,
    discount: float,
    key: PRNGKey,
    use_mean_reduction: bool,
    use_max_reduction: bool,
    convert_to_grayscale: bool = False,
) -> Tuple[PRNGKey, dict]:
    """Gets the validation metrics."""

    assert not (use_mean_reduction and use_max_reduction), "Can't use two reductions."

    key, train_key, val_key = jax.random.split(key, 3)

    # do the grayscale transform
    if convert_to_grayscale:
        val_batch = transform_to_grayscale(val_batch)

    # grab metrics for regular batch
    train_next_dist = actor(train_batch.next_observations)
    train_next_actions = train_next_dist.sample(seed=train_key)
    train_next_log_probs = train_next_dist.log_prob(train_next_actions)

    train_next_q1, train_next_q2 = target_critic(
        train_batch.next_observations, train_next_actions
    )

    if use_max_reduction:
        train_next_q = jnp.maximum(train_next_q1, train_next_q2)
    elif not use_mean_reduction:
        # not max or mean -> min
        train_next_q = jnp.minimum(train_next_q1, train_next_q2)
    else:
        train_next_q = (train_next_q1 + train_next_q2) / 2

    train_target_q = train_batch.rewards + discount * train_batch.masks * (
        train_next_q - temp() * train_next_log_probs
    )

    train_q1, train_q2 = critic(train_batch.observations, train_batch.actions)
    if use_max_reduction:
        train_q = jnp.maximum(train_q1, train_q2)
    elif not use_mean_reduction:
        train_q = jnp.minimum(train_q1, train_q2)
    else:
        train_q = (train_q1 + train_q2) / 2

    train_td_error = jnp.mean((train_q1 - train_target_q) ** 2) + jnp.mean(
        (train_q2 - train_target_q) ** 2
    )

    # grab metrics for validation batch
    val_next_dist = actor(val_batch.next_observations)
    val_next_actions = val_next_dist.sample(seed=val_key)
    val_next_log_probs = val_next_dist.log_prob(val_next_actions)

    val_next_q1, val_next_q2 = target_critic(
        val_batch.next_observations, val_next_actions
    )
    if use_max_reduction:
        val_next_q = jnp.maximum(val_next_q1, val_next_q2)
    elif not use_mean_reduction:
        val_next_q = jnp.minimum(val_next_q1, val_next_q2)
    else:
        val_next_q = (val_next_q1 + val_next_q2) / 2

    val_target_q = val_batch.rewards + discount * val_batch.masks * (
        val_next_q - temp() * val_next_log_probs
    )

    val_q1, val_q2 = critic(val_batch.observations, val_batch.actions)
    if use_max_reduction:
        val_q = jnp.maximum(val_q1, val_q2)
    elif not use_mean_reduction:
        val_q = jnp.minimum(val_q1, val_q2)
    else:
        val_q = (val_q1 + val_q2) / 2

    val_td_error = jnp.mean((val_q1 - val_target_q) ** 2) + jnp.mean(
        (val_q2 - val_target_q) ** 2
    )

    # compute o metrics
    o_phi_critic = val_td_error / train_td_error
    o_phi_actor = (
        train_next_dist.distribution.entropy().mean()
        / val_next_dist.distribution.entropy().mean()
    )
    o_phi_actor_log_prob = (
        train_next_dist.distribution.stddev().mean()
        / val_next_dist.distribution.stddev().mean()
    )

    # compute Q value means/variances
    train_q_mean, train_q_var = jnp.mean(train_q), jnp.var(train_q)
    val_q_mean, val_q_var = jnp.mean(val_q), jnp.var(val_q)

    # compute max Q values over the batch
    train_q_min, train_q_max = jnp.min(train_q), jnp.max(train_q)
    val_q_min, val_q_max = jnp.min(val_q), jnp.max(val_q)

    # return dict and rng
    return key, dict(
        o_phi_critic=o_phi_critic,
        o_phi_actor=o_phi_actor,
        o_phi_actor_log_prob=o_phi_actor_log_prob,
        train_q_mean=train_q_mean,
        train_q_var=train_q_var,
        train_q_min=train_q_min,
        train_q_max=train_q_max,
        val_q_mean=val_q_mean,
        val_q_var=val_q_var,
        val_q_min=val_q_min,
        val_q_max=val_q_max,
    )
