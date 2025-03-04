import os
import pickle
import random

import flax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from continuous_control.agents import DrQLearner
from continuous_control.datasets import Batch, ReplayBuffer
from continuous_control.evaluation import evaluate
from continuous_control.utils import make_env
from param_utils import num_params
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
flags.DEFINE_boolean("reset_only_critic", False, "Reset only the critic.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("log_wandb", False, "Whether to log to WandB.")
flags.DEFINE_boolean("save_buffer", False, "Whether to save replay buffer.")
flags.DEFINE_boolean("save_val_buffer", False, "Whether to save a validation buffer.")
flags.DEFINE_boolean(
    "saving_bad_buffer", False, "Whether we're saving a low-quality buffer."
)
flags.DEFINE_integer("n_parts", 4, "Number of parts to save of the buffer.")
flags.DEFINE_string("val_buffer_path", None, "Validation buffer path.")
flags.DEFINE_integer("validation_interval", 10_000, "When to log validation metrics.")
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

    norm_str = "layer-norm" if FLAGS.config.critic_use_layer_norm else "no-norm"
    group_name = "-".join(
        [
            FLAGS.env_name,
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
        group_name += f"-critic-hl-size-{FLAGS.config.critic_hidden_dims[0]}"
    if FLAGS.config.actor_hidden_dims[0] != 256:
        # not the same hidden layers
        group_name += f"-actor-hl-size-{FLAGS.config.actor_hidden_dims[0]}"

    if FLAGS.config.use_mean_reduction:
        group_name += "-mean-reduction"
    elif FLAGS.config.use_max_reduction:
        group_name += "-max-reduction"

    if FLAGS.resets:
        group_name += f"-reset-{FLAGS.reset_interval}"
        if FLAGS.reset_only_critic:
            group_name += "-critic-only"

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
    replay_buffer_size = kwargs.pop("replay_buffer_size")

    obs_demo = env.observation_space.sample()
    action_demo = env.action_space.sample()
    agent = DrQLearner(
        FLAGS.seed, obs_demo[np.newaxis], action_demo[np.newaxis], **kwargs
    )

    # log the number of parameters for the learner's actor
    # num_actor_params = num_params(agent.actor)
    # print("Number of actor parameters:", num_actor_params)
    # exit()

    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(
        env.observation_space,
        action_dim,
        replay_buffer_size or FLAGS.max_steps,
        n_parts=FLAGS.n_parts,
    )

    # load validation buffer in here if it exists
    if FLAGS.val_buffer_path is not None:
        assert os.path.exists(
            FLAGS.val_buffer_path
        ), f"Validation buffer path ({FLAGS.val_buffer_path}) does not exist."
        with open(FLAGS.val_buffer_path, "rb") as f:
            val_buffer = pickle.load(f)
    else:
        val_buffer = None

    # we're using an old val buffer if `v1.1` is in the name of it
    using_old_val_buffer = (
        FLAGS.val_buffer_path is not None and "v1.1" in FLAGS.val_buffer_path
    )

    # set up buffer directory if we're saving the full buffer.
    size_str = (
        "xl"
        if num_actor_convs == 4
        else "l" if num_actor_convs == 3 else "m" if num_actor_convs == 2 else "s"
    )
    buffer_dir = os.path.join(
        FLAGS.save_dir,
        f"buffers-seed-{FLAGS.seed}-{size_str}-{'og' if FLAGS.config.image_size == 84 else 'faster'}",
        (FLAGS.env_name if not FLAGS.saving_bad_buffer else f"{FLAGS.env_name}-bad"),
    )

    # find times for when we save the buffer.
    num_times_full, rem = divmod(
        FLAGS.max_steps // action_repeat, replay_buffer.capacity
    )

    # log the times when we're saving the validation buffer
    val_batches = []
    barriers = [n * FLAGS.max_steps // (10 * action_repeat) for n in range(1, 11)]

    eval_returns = []
    observation, done = env.reset(), False
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps // action_repeat + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            observation, action, reward, mask, float(done), next_observation
        )
        observation = next_observation

        if done:
            observation, done = env.reset(), False

        # if we hit `rem` or if the capacity is reached, then we save the buffer if we want to do it in the first place.
        if FLAGS.save_buffer:
            if rem != 0 and i == rem:
                replay_buffer.save_so_far(buffer_dir)
            elif i >= replay_buffer.capacity and i % replay_buffer.capacity == rem:
                # make sure that the buffer is full
                assert replay_buffer.size == replay_buffer.capacity

                round = (
                    i // replay_buffer.capacity
                )  # 1 for the first time, 2 for the second time, etc.
                assert 1 <= round <= num_times_full
                replay_buffer.save(buffer_dir, round=round)

        # if we're logging validation batches, and the number of steps is a barrier, then we log here.
        if FLAGS.save_val_buffer and (i == 1000 or i in barriers):
            # if i is 1000, we just save the first 1k -- that's easy
            if i == 1000:
                val_batch = replay_buffer.sample_first_k(1000)
            else:
                # because we're multiples of 50_000, we're either halfway in or at the end every time
                if i % 100_000 != 0:
                    # halfway in, just take from `replay_buffer.insert_index`
                    start = replay_buffer.insert_index - 1000
                    end = replay_buffer.insert_index
                    val_batch = replay_buffer.sample_between(start, end)
                else:
                    assert replay_buffer.size == replay_buffer.capacity

                    # just take from end
                    val_batch = replay_buffer.sample_last_k(1000)

            val_batches.append(val_batch)

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            update_info = agent.update(batch)

            wandb.log({f"training/{k}": v for k, v in update_info.items()})

        if i % FLAGS.validation_interval == 0 and val_buffer is not None:
            # grab validation metrics if validation buffer exists
            # we're only going to grayscale if we are using the old buffer at all
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
                use_max_reduction=FLAGS.config.use_max_reduction,
                convert_to_grayscale=using_old_val_buffer,
            )
            val_metrics = {
                f"overfitting-metrics/{k}": v for k, v in val_metrics.items()
            }
            wandb.log(val_metrics, step=i * action_repeat)

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)

            eval_returns.append((info["total"]["timesteps"], eval_stats["return"]))
            # np.savetxt(
            #     os.path.join(FLAGS.save_dir, f"{FLAGS.seed}.txt"),
            #     eval_returns,
            #     fmt=["%d", "%.1f"],
            # )

            wandb.log(
                {f"evaluation/average_{k}s": v for k, v in eval_stats.items()},
                step=info["total"]["timesteps"],
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
            # save critic encoder optimizer statistics
            old_critic_enc_opt = agent.critic.opt_state_enc

            # save actor encoder as well if we want to reset the actor.
            if FLAGS.config.decoupled:
                old_actor_enc = agent.actor.params["SharedEncoder"]
                old_actor_enc_opt = agent.actor.opt_state_enc

            # save all actor params just in case we're only resetting the critic
            old_actor_params = agent.actor.params

            # create new agent: note that the temperature is new as well
            agent = DrQLearner(
                FLAGS.seed + i,
                env.observation_space.sample()[np.newaxis],
                env.action_space.sample()[np.newaxis],
                **kwargs,
            )

            # resetting critic: copy encoder parameters and optimizer statistics
            # new_critic_params = agent.critic.params.copy(
            #     add_or_replace={"SharedEncoder": old_critic_enc}
            # )
            # agent.critic = agent.critic.replace(
            #     params=new_critic_params, opt_state_enc=old_critic_enc_opt
            # )

            # should work with dicts instead of frozendicts now
            new_critic_params = flax.core.copy(
                agent.critic.params, add_or_replace={"SharedEncoder": old_critic_enc}
            )
            agent.critic.replace(
                params=new_critic_params, opt_state_enc=old_critic_enc_opt
            )

            # resetting actor: actor in DrQ uses critic's encoder (unless it's decoupled).
            # note we could have copied enc optimizer here but actor does not affect enc (unless it's decoupled).
            if not FLAGS.reset_only_critic:
                # we are also resetting the actor.
                if not FLAGS.config.decoupled:
                    new_actor_params = flax.core.copy(
                        agent.actor.params,
                        add_or_replace={"SharedEncoder": old_critic_enc},
                    )
                    # new_actor_params = agent.actor.params.copy(
                    #     add_or_replace={"SharedEncoder": old_critic_enc}
                    # )
                    agent.actor = agent.actor.replace(params=new_actor_params)
                else:
                    new_actor_params = flax.core.copy(
                        agent.actor.params,
                        add_or_replace={"SharedEncoder": old_actor_enc},
                    )
                    # new_actor_params = agent.actor.params.copy(
                    #     add_or_replace={"SharedEncoder": old_actor_enc}
                    # )
                    agent.actor = agent.actor.replace(
                        params=new_actor_params, opt_state_enc=old_actor_enc_opt
                    )
            else:
                # we are only resetting the critic -- keep all the same actor params (aka just reset to old_actor_params)
                # TODO do we reset the opt states as well?
                agent.actor = agent.actor.replace(params=old_actor_params)

            # resetting target critic
            # new_target_critic_params = agent.target_critic.params.copy(
            #     add_or_replace={"SharedEncoder": old_target_critic_enc}
            # )
            new_target_critic_params = flax.core.copy(
                agent.target_critic.params,
                add_or_replace={"SharedEncoder": old_target_critic_enc},
            )
            agent.target_critic = agent.target_critic.replace(
                params=new_target_critic_params
            )

    # once training is done, we save the validation buffer if we want to.
    if FLAGS.save_val_buffer:
        val_buffer_dir = os.path.join(FLAGS.save_dir, "validation_buffers")
        if not os.path.exists(val_buffer_dir):
            os.makedirs(val_buffer_dir)

        # concatenate all validation batches accumulated so far.
        val_batch = Batch(
            observations=np.concatenate(
                [batch.observations for batch in val_batches], axis=0
            ),
            actions=np.concatenate([batch.actions for batch in val_batches], axis=0),
            rewards=np.concatenate([batch.rewards for batch in val_batches], axis=0),
            next_observations=np.concatenate(
                [batch.next_observations for batch in val_batches], axis=0
            ),
            masks=np.concatenate([batch.masks for batch in val_batches], axis=0),
        )
        assert val_batch.observations.shape[0] == 11_000

        buffer_save_path = os.path.join(
            val_buffer_dir, f"{FLAGS.env_name}_seed_{FLAGS.seed}.pkl"
        )
        with open(buffer_save_path, "wb") as f:
            pickle.dump(val_batch, f)


if __name__ == "__main__":
    app.run(main)
