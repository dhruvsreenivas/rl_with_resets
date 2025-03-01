# Small (S) on regular actor (XL) buffer
for seed in 4 3 2 8 9
do
    sbatch launch/run_drq_offline.sh --env_name=hopper-hop --seed=$seed --n_parts=10 --log_wandb=True --config.decoupled=True --config.actor_cnn_features="(32,)" --config.actor_cnn_strides="(2,)" --config.actor_hidden_dims="(8,8)" --max_steps=2_000_000 --replay_buffer_load_dir="/network/scratch/d/dhruv.sreenivas/drq_with_resets/buffers-seed-${seed}-xl-faster/hopper-hop" --val_buffer_path="/network/scratch/d/dhruv.sreenivas/drq_with_resets/validation_buffers/hopper-hop_seed_99.pkl"
done

for seed in 9 7 8 6 0
do
    sbatch launch/run_drq_offline.sh --env_name=pendulum-swingup --seed=$seed --n_parts=10 --log_wandb=True --config.decoupled=True --config.actor_cnn_features="(32,)" --config.actor_cnn_strides="(2,)" --config.actor_hidden_dims="(8,8)" --max_steps=2_000_000 --replay_buffer_load_dir="/network/scratch/d/dhruv.sreenivas/drq_with_resets/buffers-seed-${seed}-xl-faster/pendulum-swingup" --val_buffer_path="/network/scratch/d/dhruv.sreenivas/drq_with_resets/validation_buffers/pendulum-swingup_seed_99.pkl"
done

for seed in 7 8 9 1 0
do
    sbatch launch/run_drq_offline.sh --env_name=reacher-hard --seed=$seed --n_parts=10 --log_wandb=True --config.decoupled=True --config.actor_cnn_features="(32,)" --config.actor_cnn_strides="(2,)" --config.actor_hidden_dims="(8,8)" --max_steps=2_000_000 --replay_buffer_load_dir="/network/scratch/d/dhruv.sreenivas/drq_with_resets/buffers-seed-${seed}-xl-faster/reacher-hard" --val_buffer_path="/network/scratch/d/dhruv.sreenivas/drq_with_resets/validation_buffers/reacher-hard_seed_99.pkl"
done