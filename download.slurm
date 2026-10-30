#!/bin/bash
#SBATCH --job-name=downloader
#SBATCH --output=%x_%j.out             # Name of output file (%x expands to job name, %j to jobId)
#SBATCH --error=%x_%j.err              # Name of error file (%x expands to job name, %j to jobId)
#SBATCH --nodes=1                      # Run all processes on a single node
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=1
#SBATCH --time=200:00:00               # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                  # Number of GPUs (per node)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ji045311@ucf.edu    # Where to send mail
echo "Slurm nodes assigned: $SLURM_JOB_NODELIST"

module purge
module load anaconda/anaconda-2024.10
conda activate challenge_2026

cd /lustre/fs1/home/ji045311/challenge_2026

neuralbench emg pose -m cnn1d_pose --download

bash /home/ji045311/challenge_2026/NIPS_CHALLENGE_2026/shell/fix_emg2pose_sidecars.sh
