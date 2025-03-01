from typing import Tuple

import jax.numpy as jnp

from continuous_control.datasets import Batch
from continuous_control.networks.common import (
    InfoDict,
    Model,
    Params,
    PRNGKey,
    tree_norm,
)


def update(
    key: PRNGKey, actor: Model, critic: Model, temp: Model, batch: Batch
) -> Tuple[Model, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({"params": actor_params}, batch.observations)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp() - q).mean()
        return actor_loss, {
            "actor_loss": actor_loss,
            "entropy": -log_probs.mean(),
            "actor_pnorm": tree_norm(actor_params),
            "actor_action": jnp.mean(jnp.abs(actions)),
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    info["actor_gnorm"] = info.pop("grad_norm")

    return new_actor, info


def update_bc(key: PRNGKey, actor: Model, batch: Batch) -> Tuple[Model, InfoDict]:
    def bc_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply(
            {"params": actor_params}, batch.observations, temperature=0.0
        )

        # just take the mean/mode of the distribution -- this just requires sampling as there is no standard deviation
        actions = dist.sample(seed=key)

        # now just take MSE loss between the modes and the actions
        loss = ((actions - batch.actions) ** 2).mean()
        return loss, {"bc_loss": loss}

    new_actor, info = actor.apply_gradient(bc_loss_fn)
    info["actor_gnorm"] = info.pop("grad_norm")

    return new_actor, info
