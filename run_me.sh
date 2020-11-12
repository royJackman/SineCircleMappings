#!/usr/bin/env bash
#
#SBATCH --job-name=CAMusic
#SBATCH --time=00-3:59:59
#SBATCH --mem=8000
#SBATCH --output=./log/res_%j.txt
#SBATCH -e ./log/res_%j.err

module load python3/current
pip3 install --user numpy tensorflow pretty_midi matplotlib ffmpeg-python midiutil pygame

echo "Starting job"

python3 $HOME/CAMusic/Training.py -s -m "$HOME/CAMusic/midis/linkin_park-one_step_closer.mid" -e 100000 -w 12

exit