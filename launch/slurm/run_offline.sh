# Regular (XL) on smallest actor (S) buffer
for env_name in cheetah-run hopper-hop quadruped-run cartpole-swingup_sparse reacher-hard pendulum-swingup
do
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        sbatch launch/run_drq_offline.sh --env_name=$env_name --seed=$seed --n_parts=10 --log_wandb=True --config.decoupled=True --max_steps=2_000_000 --replay_buffer_load_dir="/network/scratch/d/dhruv.sreenivas/drq_with_resets/buffers-seed-${seed}-s-faster/${env_name}" --val_buffer_path="/network/scratch/d/dhruv.sreenivas/drq_with_resets/validation_buffers/${env_name}_seed_99.pkl"
    done
done

# Regular (XL) on medium actor (M) buffer
for env_name in cheetah-run hopper-hop quadruped-run cartpole-swingup_sparse reacher-hard pendulum-swingup
do
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        sbatch launch/run_drq_offline.sh --env_name=$env_name --seed=$seed --n_parts=10 --log_wandb=True --config.decoupled=True --max_steps=2_000_000 --replay_buffer_load_dir="/network/scratch/d/dhruv.sreenivas/drq_with_resets/buffers-seed-${seed}-m-faster/${env_name}" --val_buffer_path="/network/scratch/d/dhruv.sreenivas/drq_with_resets/validation_buffers/${env_name}_seed_99.pkl"
    done
done

# Regular (XL) on large actor (L) buffer
for env_name in cheetah-run hopper-hop quadruped-run cartpole-swingup_sparse reacher-hard pendulum-swingup
do
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        sbatch launch/run_drq_offline.sh --env_name=$env_name --seed=$seed --n_parts=10 --log_wandb=True --config.decoupled=True --max_steps=2_000_000 --replay_buffer_load_dir="/network/scratch/d/dhruv.sreenivas/drq_with_resets/buffers-seed-${seed}-l-faster/${env_name}" --val_buffer_path="/network/scratch/d/dhruv.sreenivas/drq_with_resets/validation_buffers/${env_name}_seed_99.pkl"
    done
done

# Regular (XL) on regular actor (XL) buffer
for env_name in cheetah-run hopper-hop quadruped-run cartpole-swingup_sparse reacher-hard pendulum-swingup
do
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        sbatch launch/run_drq_offline.sh --env_name=$env_name --seed=$seed --n_parts=10 --log_wandb=True --config.decoupled=True --max_steps=2_000_000 --replay_buffer_load_dir="/network/scratch/d/dhruv.sreenivas/drq_with_resets/buffers-seed-${seed}-xl-faster/${env_name}" --val_buffer_path="/network/scratch/d/dhruv.sreenivas/drq_with_resets/validation_buffers/${env_name}_seed_99.pkl"
    done
done