#!/usr/bin/env bash
#
#SBATCH --job-name=CAMusic
#SBATCH --mem=16000
#SBATCH --output=./log/res_%j.txt
#SBATCH -e ./log/res_%j.err
#SBATCH --partition=2080ti-short
#SBATCH --cpus-per-task=32
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=royjackman@umass.edu	

# --gres=gpu:4
# -p gpu
#--gres=gpu:p100:1

export CUDA_VISIBLE_DEVICES=0
source ~/miniconda3/etc/profile.d/conda.sh

conda activate py38

python3 ~/CAMusic/Training.py -b 1 -e 10000 -i 24 -m ~/CAMusic/midis/hallelujah.mid -n hallelujah-slurm -o ~/CAMusic/outputs -p 4 -r 4 -s -w 12

conda deactivate

exit