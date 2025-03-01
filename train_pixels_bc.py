import os
import pickle
import random

import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from continuous_control.agents import DrQLearner
from continuous_control.datasets import ReplayBuffer
from continuous_control.evaluation import evaluate
from continuous_control.utils import make_env
from validation_utils import get_validation_metrics

FLAGS = flags.FLAGS

flags.DEFINE_string("exp", "", "Experiment description (not actually used).")
flags.DEFINE_string("env_name", "cheetah-run", "Environment name.")
flags.DEFINE_string(
    "save_dir", "/network/scratch/d/dhruv.sreenivas/drq_with_resets", "Logging dir."
)
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 512, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(5e5), "Number of environment steps.")
flags.DEFINE_integer(
    "start_training", int(1e3), "Number of environment steps to start training."
)
flags.DEFINE_integer(
    "action_repeat", None, "Action repeat, if None, uses 2 or PlaNet default values."
)
flags.DEFINE_integer("reset_interval", 25000, "Periodicity of resets.")
flags.DEFINE_boolean("resets", False, "Periodically reset last actor / critic layers.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("log_wandb", False, "Whether to log to WandB.")
flags.DEFINE_integer("n_parts", 4, "Number of parts to save of the buffer.")
flags.DEFINE_string("val_buffer_path", None, "Validation buffer path.")
flags.DEFINE_integer("validation_interval", 10_000, "When to log validation metrics.")
flags.DEFINE_string("replay_buffer_load_dir", None, "Replay buffer loading directory.")
config_flags.DEFINE_config_file(
    "config",
    "configs/drq_original.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

PLANET_ACTION_REPEAT = {
    "cartpole-swingup": 8,
    "reacher-easy": 4,
    "cheetah-run": 4,
    "finger-spin": 2,
    "ball_in_cup-catch": 4,
    "walker-walk": 2,
}

EXPERT_INDICES = {
    "cheetah-run": 1_500_000 // 4,
    "hopper-hop": 1_500_000 // 2,
    "quadruped-run": 1_500_000 // 2,
    "pendulum-swingup": 1_500_000 // 2,
    "reacher-hard": 1_500_000 // 2,
    "cartpole-swingup_sparse": 410_000 // 2,
}


def main(_):
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    # log wandb setup
    num_critic_convs = len(FLAGS.config.critic_cnn_features)
    if FLAGS.config.actor_cnn_features is not None:
        num_actor_convs = len(FLAGS.config.actor_cnn_features)
    else:
        num_actor_convs = num_critic_convs

    if FLAGS.config.aug_actor_observations and FLAGS.config.aug_critic_observations:
        aug_str = "both-aug"
    elif FLAGS.config.aug_actor_observations:
        aug_str = "aug-actor-only"
    elif FLAGS.config.aug_critic_observations:
        aug_str = "aug-critic-only"
    else:
        aug_str = "no-aug"

    offline_buffer_str = "offline-"
    if "-s-" in FLAGS.replay_buffer_load_dir:
        offline_buffer_str += "-small"
    elif "-m-" in FLAGS.replay_buffer_load_dir:
        offline_buffer_str += "-medium"
    elif "-l-" in FLAGS.replay_buffer_load_dir:
        offline_buffer_str += "-large"
    else:
        offline_buffer_str += "-regular"

    norm_str = "layer-norm" if FLAGS.config.critic_use_layer_norm else "no-norm"
    group_name = "-".join(
        [
            FLAGS.env_name,
            "bc",
            (
                f"convs={num_critic_convs}"
                if num_critic_convs == num_actor_convs
                else f"actor_convs={num_actor_convs}-critic_convs={num_critic_convs}"
            ),
            "decoupled" if FLAGS.config.decoupled else "coupled",
            norm_str,
            aug_str,
        ]
    )
    if FLAGS.config.decoupled and not FLAGS.config.pass_grads_through_actor_encoder:
        group_name += "-random-actor-encoder"
    if not FLAGS.config.pass_grads_through_critic_encoder:
        group_name += "-random-critic-encoder"

    if FLAGS.config.critic_hidden_dims[0] != 256:
        # not the same hidden layers
        group_name += f"-critic-hidden-layer-size-{FLAGS.config.critic_hidden_dims[0]}"
    if FLAGS.config.actor_hidden_dims[0] != 256:
        # not the same hidden layers
        group_name += f"-actor-hidden-layer-size-{FLAGS.config.actor_hidden_dims[0]}"

    if FLAGS.config.use_mean_reduction:
        group_name += "-mean-reduction"

    wandb.init(
        project="dmc-drq-from-resets-repo",
        dir=FLAGS.save_dir,
        config=dict(FLAGS.config),
        group=group_name,
        mode="disabled" if not FLAGS.log_wandb else None,
    )

    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, "video", "train")
        video_eval_folder = os.path.join(FLAGS.save_dir, "video", "eval")
    else:
        video_train_folder = None
        video_eval_folder = None

    if FLAGS.action_repeat is not None:
        action_repeat = FLAGS.action_repeat
    else:
        action_repeat = PLANET_ACTION_REPEAT.get(FLAGS.env_name, 2)

    all_kwargs = FLAGS.flag_values_dict()
    all_kwargs.update(all_kwargs.pop("config"))
    kwargs = dict(FLAGS.config)

    gray_scale = kwargs.pop("gray_scale")
    image_size = kwargs.pop("image_size")

    def make_pixel_env(seed, video_folder):
        return make_env(
            FLAGS.env_name,
            seed,
            video_folder,
            action_repeat=action_repeat,
            image_size=image_size,
            frame_stack=3,
            from_pixels=True,
            gray_scale=gray_scale,
        )

    env = make_pixel_env(FLAGS.seed, video_train_folder)
    eval_env = make_pixel_env(FLAGS.seed + 42, video_eval_folder)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    assert kwargs.pop("algo") == "drq"
    kwargs.pop("replay_buffer_size")

    obs_demo = env.observation_space.sample()
    action_demo = env.action_space.sample()
    agent = DrQLearner(
        FLAGS.seed, obs_demo[np.newaxis], action_demo[np.newaxis], **kwargs
    )

    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(
        env.observation_space,
        action_dim,
        FLAGS.max_steps // action_repeat,
        n_parts=FLAGS.n_parts,
    )
    replay_buffer.load(FLAGS.replay_buffer_load_dir)

    # load validation buffer in here if it exists
    if FLAGS.val_buffer_path is not None:
        assert os.path.exists(
            FLAGS.val_buffer_path
        ), f"Validation buffer path ({FLAGS.val_buffer_path}) does not exist."
        with open(FLAGS.val_buffer_path, "rb") as f:
            val_buffer = pickle.load(f)
    else:
        val_buffer = None

    eval_returns = []
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps // action_repeat + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):

        if i >= FLAGS.start_training:
            # only sample expert data.
            batch = replay_buffer.sample_after_index(
                EXPERT_INDICES[FLAGS.env_name], FLAGS.batch_size
            )
            update_info = agent.update_bc(batch)

            wandb.log({f"training/{k}": v for k, v in update_info.items()})

        if i % FLAGS.validation_interval == 0 and val_buffer is not None:
            # grab validation metrics if validation buffer exists
            agent.rng, val_metrics = get_validation_metrics(
                agent.actor,
                agent.critic,
                agent.target_critic,
                agent.temp,
                batch,
                val_buffer,
                FLAGS.config.discount,
                agent.rng,
                use_mean_reduction=FLAGS.config.use_mean_reduction,
            )
            val_metrics = {
                f"overfitting-metrics/{k}": v for k, v in val_metrics.items()
            }
            wandb.log(val_metrics, step=i * action_repeat)

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)

            eval_returns.append((i * action_repeat, eval_stats["return"]))
            # np.savetxt(
            #     os.path.join(FLAGS.save_dir, f"{FLAGS.seed}.txt"),
            #     eval_returns,
            #     fmt=["%d", "%.1f"],
            # )

            wandb.log(
                {f"evaluation/average_{k}s": v for k, v in eval_stats.items()},
                step=i * action_repeat,
            )

        if FLAGS.resets and i % FLAGS.reset_interval == 0:
            # shared enc params: 388416
            # critic head(s) params: 366232
            # actor head params: 286882
            # so we reset roughtly half of the agent (both layer and param wise)

            # save encoder parameters
            old_critic_enc = agent.critic.params["SharedEncoder"]
            # target critic has its own copy of encoder
            old_target_critic_enc = agent.target_critic.params["SharedEncoder"]
            # save encoder optimizer statistics
            old_critic_enc_opt = agent.critic.opt_state_enc

            # create new agent: note that the temperature is new as well
            agent = DrQLearner(
                FLAGS.seed + i,
                env.observation_space.sample()[np.newaxis],
                env.action_space.sample()[np.newaxis],
                **kwargs,
            )

            # resetting critic: copy encoder parameters and optimizer statistics
            new_critic_params = agent.critic.params.copy(
                add_or_replace={"SharedEncoder": old_critic_enc}
            )
            agent.critic = agent.critic.replace(
                params=new_critic_params, opt_state_enc=old_critic_enc_opt
            )

            # resetting actor: actor in DrQ uses critic's encoder
            # note we could have copied enc optimizer here but actor does not affect enc
            new_actor_params = agent.actor.params.copy(
                add_or_replace={"SharedEncoder": old_critic_enc}
            )
            agent.actor = agent.actor.replace(params=new_actor_params)

            # resetting target critic
            new_target_critic_params = agent.target_critic.params.copy(
                add_or_replace={"SharedEncoder": old_target_critic_enc}
            )
            agent.target_critic = agent.target_critic.replace(
                params=new_target_critic_params
            )


if __name__ == "__main__":
    app.run(main)
