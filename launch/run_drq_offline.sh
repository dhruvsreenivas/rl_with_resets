#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=11:00:00
#SBATCH --requeue
#SBATCH -o /network/scratch/d/dhruv.sreenivas/drq_with_resets/job_logs_offline/output/slurm-%j.out
#SBATCH -e /network/scratch/d/dhruv.sreenivas/drq_with_resets/job_logs_offline/error/slurm-%j.err

# 1. Load your environment
module load anaconda/3
conda activate /home/mila/d/dhruv.sreenivas/anaconda3/envs/jaxrl

# 2. Launch the run.
MUJOCO_GL=egl python train_pixels_offline.py $@