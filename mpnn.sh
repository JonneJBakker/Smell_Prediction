#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=4000


module purge
module load Python/3.11.5-GCCcore-13.2.0


source $HOME/venvs/my_env/bin/activate

python main.py

deactivate