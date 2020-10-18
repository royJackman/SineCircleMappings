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
notes = range(len(chorale))

if options.graphing:
    plt.plot(notes, [float_to_note(i) for i in chorale])
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

plt.ion()
orig, pred, = plt.plot(notes, [float_to_note(i) for i in chorale], notes, [0] * len(chorale))
orig.axes.set_ylim(55, 80)
pred.axes.set_ylim(55, 80)
plt.title('Epoch 0')
plt.xlabel('Time step (in quarter-notes ♩)')
plt.ylabel('Note (in MIDI key values)')
plt.legend()
plt.grid()
plt.show(block=False)

for i in range(1, options.epochs + 1):
    x0 = np.repeat(seed[None, ...], options.batch_size, 0)
    x, loss = train_step(x0)

    step_i = len(loss_log)
    loss_log.append(loss.numpy())

    print('\r step: %d, log10(loss): %.3f'%(i+1, np.log10(loss)), end='')
    
    if step_i % options.framerate == 0:
        pred.set_ydata([float_to_note(i) for i in np.mean(x.numpy(), axis=0)[:, :, -1].flatten().tolist()[options.past_notes - 1:]])
        plt.title(f'Epoch {i - 1}')
        plt.gcf().canvas.draw()
        plt.gcf().canvas.flush_events()

suspend = input('\nPress ENTER to exit')