#!/bin/bash
#
#SBATCH --job-name=infer_sparp
#SBATCH --output=<path-to-cloned-repo>/infer_slurm_output.txt
#SBATCH --mail-user=<your-email-address>
#SBATCH --mail-type=ALL
#SBATCH --account=<your-slurm-account-type>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40GB
#SBATCH --gres=gpu:a100:1

mamba activate <your-mamba-or-conda-environment>
module purge
# module --ignore_cache load cuda/12.1
python <path-to-cloned-repo>/infer.py
