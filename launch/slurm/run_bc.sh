# Small (S) on regular actor (XL) buffer
for env_name in cheetah-run hopper-hop quadruped-run cartpole-swingup_sparse reacher-hard pendulum-swingup
do
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        sbatch launch/run_drq_bc.sh --env_name=$env_name --seed=$seed --n_parts=10 --log_wandb=True --config.decoupled=True --config.actor_cnn_features="(32,)" --config.actor_cnn_strides="(2,)" --config.actor_hidden_dims="(8,8)" --max_steps=2_000_000 --replay_buffer_load_dir="/network/scratch/d/dhruv.sreenivas/drq_with_resets/buffers-seed-${seed}-xl-faster/${env_name}"
    done
done