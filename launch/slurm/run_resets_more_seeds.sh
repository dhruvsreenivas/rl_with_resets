# S
for env_name in cheetah-run hopper-hop quadruped-run cartpole-swingup_sparse reacher-hard pendulum-swingup
do
    for seed in 10 11 12 13 14 15 16 17 18 19
    do
        sbatch launch/run_drq.sh --env_name=$env_name --seed=$seed --n_parts=10 --log_wandb=True --config.decoupled=True --config.actor_cnn_features="(32,)" --config.actor_cnn_strides="(2,)" --config.actor_hidden_dims="(8,8)" --max_steps=2_000_000 --resets=True --reset_interval=100000 --reset_only_critic=True --val_buffer_path="/network/scratch/d/dhruv.sreenivas/drq_with_resets/validation_buffers/${env_name}_seed_99.pkl"
    done
done