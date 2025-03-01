# Regular (XL)
for env_name in cartpole-swingup_sparse reacher-hard pendulum-swingup
do
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        sbatch launch/run_drq.sh --env_name=$env_name --seed=$seed --n_parts=10 --log_wandb=True --config.decoupled=True --max_steps=2_000_000 --save_buffer=True --val_buffer_path="/network/scratch/d/dhruv.sreenivas/drq_with_resets/validation_buffers/${env_name}_seed_99.pkl"
    done
done

# L
for env_name in cartpole-swingup_sparse reacher-hard pendulum-swingup
do
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        sbatch launch/run_drq.sh --env_name=$env_name --seed=$seed --n_parts=10 --log_wandb=True --config.decoupled=True --config.actor_cnn_features="(32,64,128)" --config.actor_cnn_strides="(2,2,2)" --config.actor_hidden_dims="(128,128)" --max_steps=2_000_000 --save_buffer=True --val_buffer_path="/network/scratch/d/dhruv.sreenivas/drq_with_resets/validation_buffers/${env_name}_seed_99.pkl"
    done
done

# M
for env_name in cartpole-swingup_sparse reacher-hard pendulum-swingup
do
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        sbatch launch/run_drq.sh --env_name=$env_name --seed=$seed --n_parts=10 --log_wandb=True --config.decoupled=True --config.actor_cnn_features="(32,64)" --config.actor_cnn_strides="(2,2)" --config.actor_hidden_dims="(32,32)" --max_steps=2_000_000 --save_buffer=True --val_buffer_path="/network/scratch/d/dhruv.sreenivas/drq_with_resets/validation_buffers/${env_name}_seed_99.pkl"
    done
done

# S
for env_name in cartpole-swingup_sparse reacher-hard pendulum-swingup
do
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        sbatch launch/run_drq.sh --env_name=$env_name --seed=$seed --n_parts=10 --log_wandb=True --config.decoupled=True --config.actor_cnn_features="(32,)" --config.actor_cnn_strides="(2,)" --config.actor_hidden_dims="(8,8)" --max_steps=2_000_000 --save_buffer=True --val_buffer_path="/network/scratch/d/dhruv.sreenivas/drq_with_resets/validation_buffers/${env_name}_seed_99.pkl"
    done
done