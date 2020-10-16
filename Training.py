import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from CAModel import CAModel
from DataLoader import list_chorales, stagger
import matplotlib.pyplot as plt

target = tf.pad(np.array(list_chorales[0]).astype('float32').reshape((1, -1)), [(0, 0), (15, 0)])
seed = np.zeros([1,target.shape[1],17], np.float32)
first = np.nonzero(list_chorales[0])[0]
seed[0, first[0], -1 ] = list_chorales[0][first[0]]

def loss_f(x):
    return tf.reduce_mean(tf.square(x[..., -1] - target))

ca = CAModel()

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

notes = range(len(list_chorales[0]))
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(notes, list_chorales[0])
pred, = ax2.plot(notes, [0] * len(list_chorales[0]))
plt.show()

for i in range(8000+1):
    x0 = np.repeat(seed[None, ...], 8, 0)
    x, loss = train_step(x0)

    step_i = len(loss_log)
    loss_log.append(loss.numpy())

    print('\r step: %d, log10(loss): %.3f'%(i+1, np.log10(loss)), end='')
    
    if step_i % 50 == 0:
        ax2.clear()
        ax2.plot(notes, np.mean(x.numpy(), axis=0)[:, :, -1].flatten().tolist()[15:])
        fig.canvas.draw()
        fig.canvas.flush_events()