#!/usr/bin/env bash
#
#SBATCH --job-name=CAMusic
#SBATCH --time=00-3:59:59
#SBATCH --mem=8000
#SBATCH --output=./log/res_%j.txt
#SBATCH -e ./log/res_%j.err

source ~/miniconda3/etc/profile.d/conda.sh

conda create -y --name py38 python=3.8
conda install --force-reinstall -y -q --name py38 -c conda-forge --file ~/CAMusic/requirements.txt
conda activate py38

python3 ~/CAMusic/Training.py -s -m ~/CAMusic/midis/linkin_park-one_step_closer.mid -e 100000 -w 12

conda deactivate

exit