for env_name in pendulum-swingup
do
    for seed in 99
    do
        sbatch launch/run_drq.sh --env_name=$env_name --seed=$seed --n_parts=10 --log_wandb --config.decoupled=True --max_steps=2_000_000 --save_val_buffer=True
    done
done