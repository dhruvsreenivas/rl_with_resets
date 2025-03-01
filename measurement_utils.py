from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util

from continuous_control.networks.common import Model, Params
from param_utils import l0_norm, modified_l1_norm


def get_weight_pytree(params: Params) -> Params:
    """Gets only the weight pytree (no norms or biases)."""

    flattened_params = traverse_util.flatten_dict(params, sep="/")
    return {
        k: v
        for k, v in flattened_params.items()
        if "layernorm" not in k.lower()
        and "bias" not in k.lower()
        and "spectralnorm" not in k.lower()
    }


def maybe_split_params_for_critic(params: Params) -> Params:
    vmap_critic_key = "DoubleCritic_0/VmapCritic_0/MLP_0/Dense_"
    vmap_latent_critic_key1 = "LatentDoubleCritic_0/VmapLatentCritic_0/MLP_0/Dense_"
    vmap_latent_critic_key2 = "LatentDoubleCritic_0/VmapLatentCritic_0/Dense_"

    split_params = dict()
    for k, v in params.items():
        if v.shape[0] == 2:
            # vmapcritic params
            assert (
                k.startswith(vmap_critic_key)
                or k.startswith(vmap_latent_critic_key1)
                or k.startswith(vmap_latent_critic_key2)
            )

            # number for the layer itself (0, 2, 4 for first critic, 1, 3, 5 for second critic)
            if k.startswith(vmap_critic_key):
                num = int(k[len(vmap_critic_key)])
            elif k.startswith(vmap_latent_critic_key1):
                num = int(k[len(vmap_latent_critic_key1)])
            else:
                num = int(k[len(vmap_latent_critic_key2)])

            first_critic_params = v[0]
            second_critic_params = v[1]

            if k.startswith(vmap_critic_key):
                first_k = vmap_critic_key + str(2 * num) + k[len(vmap_critic_key) + 1 :]
                second_k = (
                    vmap_critic_key + str(2 * num + 1) + k[len(vmap_critic_key) + 1 :]
                )
            elif k.startswith(vmap_latent_critic_key1):
                first_k = (
                    vmap_latent_critic_key1
                    + str(2 * num)
                    + k[len(vmap_latent_critic_key1) + 1 :]
                )
                second_k = (
                    vmap_latent_critic_key1
                    + str(2 * num + 1)
                    + k[len(vmap_latent_critic_key1) + 1 :]
                )
            else:
                first_k = (
                    vmap_latent_critic_key2
                    + str(2 * num)
                    + k[len(vmap_latent_critic_key2) + 1 :]
                )
                second_k = (
                    vmap_latent_critic_key2
                    + str(2 * num + 1)
                    + k[len(vmap_latent_critic_key2) + 1 :]
                )

            split_params[first_k] = first_critic_params
            split_params[second_k] = second_critic_params
        else:
            split_params[k] = v

    # we have that first critic to be {NET_NAME}_{0, 2, 4}, second critic to be {NET_NAME}_{1, 3, 5}, as this is how we log activation ranks.
    return split_params


def compute_ranks(activation: jnp.ndarray) -> dict:
    svds = jnp.linalg.svdvals(activation)
    implicit_rank = (svds > 1e-6).sum()

    frob_norm = (svds**2).sum()
    stable_rank = frob_norm / (svds[0] ** 2)

    cum_energy = jnp.cumsum(svds, axis=0) / svds.sum()
    effective_rank = (cum_energy < 0.99).sum() + 1

    # compute the number of dead activations
    dead_units = (activation.sum(axis=0) == 0).sum()
    dead_units_prop = dead_units / activation.shape[1]

    return dict(
        implicit_rank=implicit_rank,
        stable_rank=stable_rank,
        effective_rank=effective_rank,
        dead_units=dead_units,
        dead_units_prop=dead_units_prop,
    )


def gather_metrics(
    model: Model,
    init_params: Params,
    activations: List[jnp.ndarray],
    grads: Optional[Params],
) -> dict:
    """Gather all metrics for this particular model."""

    weight_params = get_weight_pytree(model.params)
    init_weight_params = get_weight_pytree(init_params)

    # split for critic just in case (here, we split such that the DoubleCritic_0/VmapCritic_0/Dense_{0, 2, 4} are first critic, DoubleCritic_0/VmapCritic_0/Dense_{1, 3, 5} are for second critic)
    # done only for comparison purposes
    weight_params = maybe_split_params_for_critic(weight_params)
    init_weight_params = maybe_split_params_for_critic(init_weight_params)

    # now compute the L1/L2/L0 norm of all weights (all still dicts, key for each relevant param, with leaves of size 1)
    weight_l0_norms = jax.tree_util.tree_map(
        lambda param: l0_norm(param) / np.prod(param.shape),
        weight_params,
    )
    weight_l1_norms = jax.tree_util.tree_map(
        lambda param: modified_l1_norm(param) / np.prod(param.shape),
        weight_params,
    )
    weight_l2_norms = jax.tree_util.tree_map(
        lambda param: jnp.linalg.norm(param), weight_params
    )

    implicit_ranks = []
    stable_ranks = []
    effective_ranks = []
    dead_units_props = []
    total_dead_units = 0
    total_activations = 0

    for activation in activations:
        # compute ranks given activations
        if activation.ndim == 4:
            activation = jnp.reshape(activation, (activation.shape[0], -1))
        elif activation.ndim == 3:
            assert activation.shape[0] == 2
            first_critic_activation = activation[0]
            second_critic_activation = activation[1]

        # now compute the ranks
        if activation.ndim != 3:
            rank_dict = compute_ranks(activation)

            implicit_ranks.append(rank_dict["implicit_rank"])
            stable_ranks.append(rank_dict["stable_rank"])
            effective_ranks.append(rank_dict["effective_rank"])
            dead_units_props.append(rank_dict["dead_units_prop"])

            total_dead_units += rank_dict["dead_units"]
            total_activations += activation.shape[1]
        else:
            for activ in [first_critic_activation, second_critic_activation]:
                rank_dict = compute_ranks(activ)

                implicit_ranks.append(rank_dict["implicit_rank"])
                stable_ranks.append(rank_dict["stable_rank"])
                effective_ranks.append(rank_dict["effective_rank"])
                dead_units_props.append(rank_dict["dead_units_prop"])

                total_dead_units += rank_dict["dead_units"]
                total_activations += activ.shape[1]

    # now compute the L1/L2/L0 norm of all grads if they exist (all still dicts, each with leaves of size 1)
    if grads is not None:
        weight_grads = get_weight_pytree(grads)
        weight_grads = maybe_split_params_for_critic(weight_grads)

        grad_l0_norms = jax.tree_util.tree_map(
            lambda param: l0_norm(param) / np.prod(param.shape),
            weight_grads,
        )
        grad_l1_norms = jax.tree_util.tree_map(
            lambda param: modified_l1_norm(param) / np.prod(param.shape),
            weight_grads,
        )
        grad_l2_norms = jax.tree_util.tree_map(
            lambda param: jnp.linalg.norm(param), weight_grads
        )
    else:
        grad_l0_norms, grad_l1_norms, grad_l2_norms = None, None, None

    # dict with each diff per layer
    dist_from_init = jax.tree_util.tree_map(
        lambda x, y: jnp.linalg.norm(x - y), weight_params, init_weight_params
    )
    num_params = jax.tree_util.tree_map(
        lambda param: jnp.sqrt(np.prod(param.shape)), weight_params
    )
    dist_from_init = jax.tree_util.tree_map(
        lambda x, y: x / y, dist_from_init, num_params
    )

    # divide by sqrt(params) for weight l2 norm + grad l2 norm as well
    weight_l2_norms = jax.tree_util.tree_map(
        lambda x, y: x / y, weight_l2_norms, num_params
    )
    if grad_l2_norms is not None:
        grad_l2_norms = jax.tree_util.tree_map(
            lambda x, y: x / y, grad_l2_norms, num_params
        )

    return dict(
        dist_from_init=dist_from_init,
        num_params=num_params,
        grad_l0_norms=grad_l0_norms,
        grad_l1_norms=grad_l1_norms,
        grad_l2_norms=grad_l2_norms,
        weight_l0_norms=weight_l0_norms,
        weight_l1_norms=weight_l1_norms,
        weight_l2_norms=weight_l2_norms,
        dead_units_props=dead_units_props,  # per activation
        total_dead_prop=total_dead_units / total_activations,
        implicit_ranks=implicit_ranks,  # per activation
        stable_ranks=stable_ranks,
        effective_ranks=effective_ranks,
    )


def organize_for_logging(metrics: dict, from_critic: bool = False) -> dict:
    """Organizes the metrics to log to wandb."""

    # set up logging dict
    log_dict = dict()
    param_keys = list(metrics["dist_from_init"].keys())

    # for all lists in the metrics dict, convert them to dicts with the right keys.
    for k, v in metrics.items():
        if (
            isinstance(v, dict)
            or isinstance(v, jnp.ndarray)
            or isinstance(v, float)
            or isinstance(v, int)
        ):
            log_dict[k] = v
        else:
            # list
            assert isinstance(v, list)
            list2dict = {param_keys[i]: v[i] for i in range(len(param_keys))}
            log_dict[k] = list2dict

    # flatten dict with `flax.traverse_util`
    log_dict = traverse_util.flatten_dict(log_dict, sep="/")

    # if from critic, add `critic-metric-logging/` to each of the keys, else `actor-metric-logging/`
    if from_critic:
        log_dict = {f"critic-metric-logging/{k}": v for k, v in log_dict.items()}
    else:
        log_dict = {f"actor-metric-logging/{k}": v for k, v in log_dict.items()}

    for k, v in log_dict.items():
        assert isinstance(v, jax.Array)

    return log_dict
