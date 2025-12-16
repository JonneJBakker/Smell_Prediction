#!/bin/bash
#SBATCH --time=00:06:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=4000


module purge
module load Python/3.11.5-GCCcore-13.2.0
module load SciPy-bundle/2023.11-gfbf-2023b

source $HOME/venvs/cBERTa/bin/activate

python main.py

deactivate