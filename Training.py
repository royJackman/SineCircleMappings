import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf

from CAModel import CAModel
from DataLoader import list_chorales, float_to_note
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-b', '--batch_size', type=int, dest='batch_size', default=8, help='Set batch size')
parser.add_option('-c', '--chorale', type=int, dest='chorale', default=0, help='Which chorale to use as a model')
parser.add_option('-e', '--epochs', type=int, dest='epochs', default=8000, help='Number of learning epochs')
parser.add_option('-f', '--framerate', type=int, dest='framerate', default=20, help='Number of epochs between graph updates')
parser.add_option('-g', '--graphing', action='store_true', dest='graphing', default=False, help='Print chorale and exit')
parser.add_option('-p', '--past-notes', type=int, dest='past_notes', default=16, help='How far into the past to stretch the convolutional window')
(options, args) = parser.parse_args()

chorale = list_chorales[options.chorale]
note_chorale = [float_to_note(i) for i in chorale]
notes = range(len(chorale))

if options.graphing:
    plt.plot(notes, note_chorale)
    plt.title(f'Chorale {options.chorale}')
    plt.xlabel('Time step (in quarter-notes ♩)')
    plt.ylabel('Note (in MIDI key values)')
    plt.ylim(55, 80)
    plt.show()
    sys.exit(f'Graphing of chorale {options.chorale} complete! Exiting..')

target = tf.pad(np.array(chorale).astype('float32').reshape((1, -1)), [(0, 0), (options.past_notes - 1, 0)])
seed = np.zeros([1,target.shape[1],options.past_notes + 1], np.float32)
first = np.nonzero(chorale)[0]
seed[0, first[0], -1] = chorale[first[0]]

def loss_f(x): return tf.reduce_mean(tf.square(x[..., -1] - target))

ca = CAModel(past_notes=options.past_notes)

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
root = np.sqrt(options.batch_size + 1)
rows = np.floor(root)
cols = np.ceil(root)
if rows * cols < options.batch_size + 1:
    rows += 1
fig, axs = plt.subplots(int(rows), int(cols), sharex=True, sharey=True)
plt.setp(axs, ylim=(55, 80))
for i, a in enumerate(np.asarray(axs).flatten()):
    if i == 0:
        a.set_title('Average')
    elif i < options.batch_size + 1:
        a.set_title(f'Batch {i}')
    else:
        fig.delaxes(a)
    if i < options.batch_size + 1:
        a.plot(notes, note_chorale)
        lines.append(a.plot(notes, [0] * len(chorale))[0])
fig.suptitle('Epoch 0')
fig.text(0.5, 0.04, 'Time step (in quarter-notes ♩)', ha='center')
fig.text(0.04, 0.5, 'Note (in MIDI key values)', va='center', rotation='vertical')
mgr = plt.get_current_fig_manager().window.state('zoomed')
plt.show()

for i in range(1, options.epochs + 1):
    x0 = np.repeat(seed[None, ...], options.batch_size, 0)
    x, loss = train_step(x0)

    step_i = len(loss_log)
    loss_log.append(loss.numpy())

    print('\r step: %d, log10(loss): %.3f'%(i+1, np.log10(loss)), end='')
    
    if step_i % options.framerate == 0:
        xn = x.numpy()
        lines[0].set_ydata([float_to_note(j) for j in np.mean(xn, axis=0)[:, :, -1].flatten().tolist()[options.past_notes - 1:]])
        for key, val in enumerate(lines[1:]):
            val.set_ydata([float_to_note(j) for j in xn[key, :, :, -1].flatten().tolist()[options.past_notes - 1:]])
        fig.suptitle(f'Epoch {i - 1}')
        plt.gcf().canvas.draw()
        plt.gcf().canvas.flush_events()

suspend = input('\nPress ENTER to exit')