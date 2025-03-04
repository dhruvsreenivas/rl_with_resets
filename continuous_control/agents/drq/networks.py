from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

from continuous_control.networks.common import default_init
from continuous_control.networks.critic_net import DoubleCritic
from continuous_control.networks.policies import NormalTanhPolicy


class Encoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = "VALID"

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        x = observations.astype(jnp.float32) / 255.0
        for features, stride in zip(self.features, self.strides):
            x = nn.Conv(
                features,
                kernel_size=(3, 3),
                strides=(stride, stride),
                kernel_init=default_init(),
                padding=self.padding,
            )(x)
            x = nn.relu(x)

        if len(x.shape) == 4:
            x = x.reshape([x.shape[0], -1])
        else:
            x = x.reshape([-1])
        return x


class DrQDoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = "VALID"
    latent_dim: int = 50
    use_layer_norm: bool = False
    pass_grads_through_encoder: bool = True

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = Encoder(
            self.cnn_features, self.cnn_strides, self.cnn_padding, name="SharedEncoder"
        )(observations)

        # If we want to stop gradients through the encoder we do so here.
        if not self.pass_grads_through_encoder:
            x = jax.lax.stop_gradient(x)

        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        return DoubleCritic(self.hidden_dims, use_layer_norm=self.use_layer_norm)(
            x, actions
        )


class DrQPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = "VALID"
    latent_dim: int = 50
    pass_grads_through_encoder: bool = False

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0
    ) -> tfd.Distribution:
        x = Encoder(
            self.cnn_features, self.cnn_strides, self.cnn_padding, name="SharedEncoder"
        )(observations)

        # If we want to stop gradients through the encoder we do so here.
        if not self.pass_grads_through_encoder:
            x = jax.lax.stop_gradient(x)

        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        return NormalTanhPolicy(self.hidden_dims, self.action_dim)(x, temperature)
