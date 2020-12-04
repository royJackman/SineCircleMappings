import tensorflow as tf
import numpy as np

tf.config.run_functions_eagerly(True)

class CAModel(tf.keras.Model):
    def __init__(self, past_notes=16, width=1, filters=128, piano_roll=False):
        super().__init__()
        self.past_notes = past_notes
        self.width = width
        self.filters = filters
        self.piano_roll = piano_roll

        layers = [tf.keras.layers.MaxPool2D(pool_size=(3,3), padding='same', strides=(1,1))] if self.piano_roll else []
        layers.append(tf.keras.layers.Conv2D(self.filters, 1, activation=tf.nn.relu))
        layers.append(tf.keras.layers.Conv2D(self.past_notes + 1, 1, activation=None, kernel_initializer=tf.keras.initializers.zeros()))

        self.dmodel = tf.keras.Sequential(layers)

        self(tf.zeros([1, 128 if self.piano_roll else self.width, self.past_notes, self.past_notes + 1]))

    @tf.function
    def perceive(self, x):
        identity = np.zeros((self.width, self.past_notes)).astype('float32')
        identity[0, -1] = 1.0
        uniform = (1.0/self.past_notes) * np.ones((self.width, self.past_notes)).astype('float32')
        linear = np.linspace(0.0, 1.0, self.past_notes + 1)[1:].astype('float32')
        linear /= linear.sum()
        linear = np.tile(linear, (self.width, 1))
        log = np.logspace(0.0, 1.0, self.past_notes + 1)[1:].astype('float32')
        log /= log.sum()
        log = np.tile(log, (self.width, 1))
        kernel = tf.stack([identity, uniform, linear, log], -1)[:, :, None, :]
        kernel = tf.repeat(kernel, self.past_notes + 1, 2)
        y = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], 'SAME')
        return y

    @tf.function
    def call(self, x, fire_rate=0.5, step_size=1.0):
        y = self.perceive(x)
        dx = self.dmodel(y)*step_size
        update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate
        x += dx * tf.cast(update_mask, tf.float32)
        return x

    # @tf.function
    # def call(self, x, step_size=1.0):
    #     y = self.percieve(x)
    #     dx = self.dmodel(y) * step_size
    #     curr_theta =  x.numpy()[:, :, :, -1]
    #     x += dx
    #     npx = x.numpy()
    #     npx[:, :, :, -1] = curr_theta + npx[:, :, :, -3] + npx[:, :, :, -2] * ((-1/2)*np.sin(curr_theta * (2*np.pi)))
    #     x = tf.convert_to_tensor(npx)
    #     return x

    # @tf.function
    # def perceive(self, x, angle=0.0):
    #     identify = np.float32([0, 1, 0])
    #     identify = np.outer(identify, identify)
    #     dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
    #     dy = dx.T
    #     c, s = tf.cos(angle), tf.sin(angle)
    #     kernel = tf.stack([identify, c*dx-s*dy, s*dx+c*dy], -1)[:, :, None, :]
    #     kernel = tf.repeat(kernel, self.channel_n, 2)
    #     print(kernel.shape)
    #     y = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], 'SAME')
    #     return y
