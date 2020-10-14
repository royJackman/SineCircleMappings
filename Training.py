import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from CAModel import CAModel
from DataLoader import list_chorales, stagger
import matplotlib.pyplot as plt

target = tf.pad(np.array(list_chorales[0]).astype('float32').reshape((1, -1)), [(0, 0), (15, 0)])
seed = np.zeros([1,target.shape[1],19], np.float32)
first = np.nonzero(list_chorales[0])[0]
seed[0, first[0]] = list_chorales[0][first[0]]

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

for i in range(8000+1):
    x0 = np.repeat(seed[None, ...], 8, 0)
    x, loss = train_step(x0)
    step_i = len(loss_log)
    loss_log.append(loss.numpy())

    print('\r step: %d, log10(loss): %.3f'%(len(loss_log), np.log10(loss)), end='')