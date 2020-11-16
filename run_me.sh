#!/usr/bin/env bash
#
#SBATCH --job-name=CAMusic
#SBATCH --mem=8000
#SBATCH --output=./log/res_%j.txt
#SBATCH -e ./log/res_%j.err
#SBATCH --partition=2080ti-long
#SBATCH --cpus-per-task=32

source ~/miniconda3/etc/profile.d/conda.sh

conda activate py38

python3 ~/CAMusic/Training.py -s -o ~/CAMusic/outputs -m ~/CAMusic/midis/linkin_park-one_step_closer.mid -e 1000 -w 12

conda deactivate

exit