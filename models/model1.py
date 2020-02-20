import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, BatchNormalization, Activation, GRU, Flatten


class Model1(tf.keras.Sequential):
    def __init__(self, record_size):
        super().__init__([
            Input(shape=(record_size, 1)),

            Conv1D(32, 9, strides=4),
            BatchNormalization(),
            Activation('relu'),

            Conv1D(64, 9, strides=4),
            BatchNormalization(),
            Activation('relu'),

            Conv1D(128, 9, strides=4),
            BatchNormalization(),
            Activation('relu'),

            GRU(128),

            Flatten(),
            Dense(4, activation='softmax')
        ])
