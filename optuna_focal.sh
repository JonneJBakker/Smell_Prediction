#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=3000


module purge
module load Python/3.11.5-GCCcore-13.2.0
module load SciPy-bundle/2023.11-gfbf-2023b

source $HOME/venvs/cBERTa/bin/activate

python optuna_focal_loss.py --train_csv Data/splits/train_stratified80.csv --val_csv Data/splits/val_stratified10.csv --smiles_column nonStereoSMILES --output_dir optuna_asym_focal --n_trials 20

deactivate