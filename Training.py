import argparse
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf

from CAModel import CAModel
from DataLoader import list_chorales, float_to_note
from MIDIConverter import midi_to_chroma

parser = argparse.ArgumentParser('Train a model on a midi file')
parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=8, help='Set batch size')
parser.add_argument('-c', '--chorale', type=int, dest='chorale', default=0, help='Which chorale to use as a model')
parser.add_argument('-e', '--epochs', type=int, dest='epochs', default=8000, help='Number of learning epochs')
parser.add_argument('-f', '--framerate', type=int, dest='framerate', default=20, help='Number of epochs between graph updates')
parser.add_argument('-g', '--graphing', action='store_true', dest='graphing', default=False, help='Print chorale and exit')
parser.add_argument('-m', '--midi-file', type=str, dest='midi_file', default=None, help='MIDI file to process, will override chorale')
parser.add_argument('-p', '--past-notes', type=int, dest='past_notes', default=16, help='How far into the past to stretch the convolutional window')
parser.add_argument('-r', '--chroma-frequency', type=int, dest='chroma_frequency', default=4, help='MIDI to chroma sampling frequency')
parser.add_argument('-s', '--slurm', action='store_true', dest='slurm', default=False, help='Just the learning')
parser.add_argument('-w', '--width', type=int, dest='width', default=1, help='The width of the convolutional window, how many other notes the model can see')
args = parser.parse_args()

if args.midi_file is None:
    chorale = list_chorales[args.chorale]
    note_chorale = [float_to_note(i) for i in chorale]
    notes = range(len(chorale))
    chorale = np.array(chorale).reshape((1, -1))
else:
    chorale = midi_to_chroma(args.midi_file, fs=args.chroma_frequency)
    note_chorale = (chorale - np.min(chorale))/(np.max(chorale) - np.min(chorale))
    notes = range(note_chorale.shape[1])

if args.graphing:
    if note_chorale is list:
        plt.plot(notes, note_chorale)
        plt.title(f'Chorale {args.chorale}')
        plt.ylim(55, 80)
        plt.xlabel('Time step')
        plt.ylabel('Note')
    else:
        fig, axs = plt.subplots(3, 4, sharex=True, sharey=True)
        for i, a in enumerate(np.asarray(axs).flatten()):
            a.plot(notes, note_chorale[i].tolist())
        fig.suptitle(args.midi_file)
        fig.text(0.5, 0.04, 'Time step', ha='center')
        fig.text(0.04, 0.5, 'Note', va='center', rotation='vertical')
    plt.show()
    sys.exit('Graphing complete! Exiting..')

target = tf.pad(np.array(note_chorale).astype('float32'), [(0, 0), (args.past_notes - 1, 0)])
seed = np.zeros([target.shape[0],target.shape[1],args.past_notes + 1], np.float32)
seed[:, args.past_notes-1, -1] = note_chorale[:, 0]

def loss_f(x): return tf.reduce_mean(tf.square(x[..., -1] - target))
def scale(x): return x if args.midi_file is not None else float_to_note(x)

ca = CAModel(past_notes=args.past_notes, width=args.width)

loss_log = []

lr = 2e-3
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay([2000], [lr, lr * 0.1])
trainer = tf.keras.optimizers.Adam(lr_sched)

loss0 = loss_f(seed).numpy()

@tf.function
def train_step(x):
    iter_n = tf.random.uniform([], 64, 96, tf.int32)
    with tf.GradientTape() as g:
        for i in tf.range(iter_n):
            x = ca(x)
        loss = tf.reduce_mean(loss_f(x))
    grads = g.gradient(loss, ca.weights, unconnected_gradients='zero')
    grads = [g/(tf.norm(g)+1e-8) for g in grads]
    trainer.apply_gradients(zip(grads, ca.weights))
    return x, loss

lines = []
plt.ion()
plt.rcParams['axes.grid'] = True
music_graphs = 1 if args.midi_file is None else 12
batch_graphs = args.batch_size if args.midi_file is None else 0
total_graphs = music_graphs + batch_graphs
root = np.sqrt(total_graphs)
rows = np.floor(root)
cols = np.ceil(root)
if rows * cols < total_graphs:
    rows += 1
fig, axs = plt.subplots(int(rows), int(cols), sharex=True, sharey=True)
for i, a in enumerate(np.asarray(axs).flatten()):
    if i < music_graphs:
        a.set_title(f'Music Channel {i + 1}')
        a.plot(notes, note_chorale[i])
        lines.append(a.plot(notes, [0] * max(chorale.shape))[0])
    elif i < total_graphs:
        a.set_title(f'Batch {i - music_graphs + 1}')
        a.plot(notes, np.mean(note_chorale, axis=0))
        lines.append(a.plot(notes, [0] * max(chorale.shape))[0])
    else:
        fig.delaxes(a)
fig.suptitle('Epoch 0')
fig.text(0.5, 0.04, 'Time step', ha='center')
fig.text(0.04, 0.5, 'Note', va='center', rotation='vertical')
mgr = plt.get_current_fig_manager().window.state('zoomed')

if not args.slurm:
    plt.show()

framenum = 0

for i in range(1, args.epochs + 1):
    x0 = np.repeat(seed[None, ...], args.batch_size, 0)
    x, loss = train_step(x0)

    step_i = len(loss_log)
    loss_log.append(loss.numpy())

    print('\r step: %d, log10(loss): %.3f'%(i+1, np.log10(loss)), end='')
    
    if step_i % args.framerate == 0:
        xn = x.numpy()
        for j in range(music_graphs):
            lines[j].set_ydata([scale(k) for k in np.mean(xn, axis=0)[j, :, -1].flatten().tolist()[args.past_notes - 1:]])
        for j in range(batch_graphs):
            lines[music_graphs + j].set_ydata([scale(k) for k in np.mean(xn, axis=1)[j, :, -1].flatten().tolist()[args.past_notes - 1:]])
        fig.suptitle(f'Epoch {i - 1}')
        plt.gcf().canvas.draw()
        plt.gcf().canvas.flush_events()
        plt.savefig(f'./outputs/epoch-{framenum}.jpg')
        framenum += 1

if not args.slurm:
    ffmpeg.input('/outputs/*.jpg', framerate=25).output('output.gif').run()
    suspend = input('\nPress ENTER to exit')