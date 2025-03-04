import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.algo = "sac"

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.tau = 0.005
    config.target_update_period = 1
    config.updates_per_step = 1

    config.init_temperature = 1.0
    config.target_entropy = None
    config.use_mean_reduction = False

    config.replay_buffer_size = int(1e6)

    return config
