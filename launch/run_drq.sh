#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=18:45:00
#SBATCH --requeue
#SBATCH -o /network/scratch/d/dhruv.sreenivas/drq_with_resets/job_logs/output/slurm-%j.out
#SBATCH -e /network/scratch/d/dhruv.sreenivas/drq_with_resets/job_logs/error/slurm-%j.err

# 1. Load your environment
module load anaconda/3
conda activate /home/mila/d/dhruv.sreenivas/anaconda3/envs/jaxrl

# 2. Launch the run.
MUJOCO_GL=egl python train_pixels.py $@