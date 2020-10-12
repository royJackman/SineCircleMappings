import tensorflow as tf
import numpy as np

tf.config.run_functions_eagerly(True)

class CAModel(tf.keras.Model):
    def __init__(self, channels=19, past_notes=16):
        super().__init__()
        self.channels = channels
        self.past_notes = past_notes

        self.dmodel = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, 1, activation=tf.nn.relu),
            tf.keras.layers.Conv2D(self.channels, 1, activation=None, kernel_initializer=tf.zeros_initializer)
        ])

        self(tf.zeros([1, 1, self.past_notes, channels]))

    @tf.function
    def percieve(self, x):
        identity = np.zeros(self.past_notes).reshape((1, -1)).astype('float32')
        identity[-1] = 1.0
        uniform = (1.0/self.past_notes) * np.ones(self.past_notes).reshape((1, -1)).astype('float32')
        linear = np.linspace(0.0, 1.0, self.past_notes + 1)[1:].reshape((1, -1)).astype('float32')
        log = np.logspace(0.0, 1.0, self.past_notes + 1)[1:].reshape((1, -1)).astype('float32')
        kernel = tf.stack([identity, uniform, linear, log], -1)[:, :, None, :]
        kernel = tf.repeat(kernel, self.channels, 2)
        y = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], 'SAME')
        return y

    @tf.function
    def call(self, x, step_size=1.0):
        y = self.percieve(x)
        dx = self.dmodel(y) * step_size
        curr_theta =  x.numpy()[:, :, -1, :]
        x += dx
        npx = x.numpy()
        npx[:, :, -1, :] = curr_theta + npx[:, :, -3, :] + npx[:, :, -2, :] * ((-1/2)*np.sin(curr_theta * (2*np.pi)))
        x = tf.convert_to_tensor(npx)
        return x
