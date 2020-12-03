import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from CAModel import CAModel
from MIDIConverter import midi_to_chroma, midi_to_piano_roll
from tqdm import tqdm

parser = argparse.ArgumentParser('Test a model on a midi file')
parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=8, help='Set batch size')
parser.add_argument('-c', '--chroma-frequency', type=int, dest='chroma_frequency', default=4, help='MIDI to chroma sampling frequency')
parser.add_argument('-l', '--load-model', type=str, dest='model', required=True)
parser.add_argument('-m', '--midi-file', type=str, dest='midi_file', required=True, help='The midi file to test the model on')
parser.add_argument('-p', '--piano-roll', action='store_true', dest='piano_roll', default=False, help='Use piano roll instead of chromagraph')
args = parser.parse_args()


loss_log = []

ca = CAModel(past_notes=4, width=12, filters=24)
ca.load_weights(args.model)

chorale = midi_to_piano_roll(args.midi_file, fs=args.chroma_frequency) if args.piano_roll else midi_to_chroma(args.midi_file, fs=args.chroma_frequency)
note_chorale = (chorale - np.min(chorale))/(np.max(chorale) - np.min(chorale))

target = tf.pad(np.array(note_chorale).astype('float32'), [(0, 0), (4 - 1, 0)])
seed = np.zeros([target.shape[0],target.shape[1],4 + 1], np.float32)
seed[:, 4-1, -1] = note_chorale[:, 0]

def loss_f(x): return tf.reduce_mean(tf.square(x[..., -1] - target))

x0 = np.repeat(seed[None, ...], args.batch_size, 0)

plt.ion()
if args.piano_roll:
    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(target, aspect='auto')
    axs[1].imshow(tf.reduce_mean(x0[..., -1], 0), aspect='auto')
plt.show()

for i in tqdm(range(80)):
    x0 = ca(x0)
    loss_log.append(np.log10(tf.reduce_mean(loss_f(x0))))
    axs[1].imshow(tf.reduce_mean(x0[..., -1], 0), aspect='auto')
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()

input('Press ENTER to exit')