import jax
import jax.numpy as jnp
import optax

from continuous_control.networks.common import Model, Params


def l0_norm(arr: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(arr != 0.0)


def modified_l1_norm(arr: jnp.ndarray) -> jnp.ndarray:
    if arr.ndim <= 2:
        return jnp.linalg.norm(arr, ord=1)

    return jnp.abs(arr).sum()


def param_l2_diff(first_params: Params, second_params: Params) -> jnp.ndarray:
    param_diff = jax.tree_util.tree_map(lambda x, y: x - y, first_params, second_params)
    return optax.global_norm(param_diff)


def num_params(model: Model) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(model.params))
