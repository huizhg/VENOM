#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=Project_2005312
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:a100:1,nvme:950
#SBATCH --output momentum_05_final_result.txt
#SBATCH --error test_err.txt
#$LOCAL_SCRATCH


module load gcc/10.4.0 cuda/12.1.1
export PATH="/scratch/project_2007291/conda/diff/bin:$PATH"
srun nvidia-smi

python3 run_venom_nae.py --test True --save_dir ./images/no_input_target