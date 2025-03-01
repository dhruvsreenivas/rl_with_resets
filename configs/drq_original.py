import ml_collections
from ml_collections import config_dict


def get_config():
    config = ml_collections.ConfigDict()

    config.algo = "drq"

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.critic_hidden_dims = (256, 256)
    config.actor_hidden_dims = (256, 256)

    config.critic_cnn_features = (32, 64, 128, 256)
    config.critic_cnn_strides = (2, 2, 2, 2)
    config.actor_cnn_features = config_dict.placeholder(tuple)
    config.actor_cnn_strides = config_dict.placeholder(tuple)
    config.cnn_padding = "SAME"
    config.latent_dim = 50

    config.discount = 0.99

    config.tau = 0.005
    config.target_update_period = 1

    config.init_temperature = 0.1
    config.target_entropy = None

    config.decoupled = False
    config.pass_grads_through_actor_encoder = True
    config.pass_grads_through_critic_encoder = True
    config.critic_use_layer_norm = False
    config.aug_actor_observations = True
    config.aug_critic_observations = True
    config.use_mean_reduction = False
    config.use_max_reduction = False

    config.replay_buffer_size = 100_000

    config.gray_scale = True
    config.image_size = 64

    return config
