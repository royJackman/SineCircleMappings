import argparse
import numpy as np
import matplotlib.pyplot as plt

from mido import MidiFile\

parser = argparse.ArgumentParser('Convert MIDI files to numpy')
parser.add_argument('-i', '--input-file', dest='input_filename', type=str, help='File to read from', required=True)
parser.add_argument('-o', '--output-file', dest='output_filename', type=str, default='output', help='File to save output')
args = parser.parse_args()

retval = []
tempos = []
mid = MidiFile(args.input_filename)
for i, track in enumerate(mid.tracks):
    t = []
    print(f'Track {i}: {track.name}')
    time = 0
    for msg in track:
        if msg.is_meta and msg.type == 'set_tempo':
            tempos.append((msg.tempo, msg.time))
        if msg.type == 'note_on':
            t += [msg.note] * int(np.ceil(msg.time / mid.ticks_per_beat))
    retval.append(t)

root = int(np.ceil(np.sqrt(len(mid.tracks))))
fig, axs = plt.subplots(root, root)

for i, a in enumerate(np.asarray(axs).flatten()):
    if i < len(mid.tracks) and len(retval[i]) > 0:
        a.plot(retval[i])
    else:
        fig.delaxes(a)
plt.show()