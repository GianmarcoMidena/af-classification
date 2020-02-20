import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, BatchNormalization, Activation, GRU, Bidirectional, Flatten, \
                                    MaxPool1D


class Model2(tf.keras.Sequential):
    def __init__(self, record_size):
        super().__init__([
            Input(shape=(record_size, 1)),

            Conv1D(64, 9, strides=2),
            BatchNormalization(),
            Activation('relu'),

            Conv1D(64, 9, strides=2),
            BatchNormalization(),
            Activation('relu'),
            MaxPool1D(2, 2),

            Conv1D(128, 9, strides=2),
            BatchNormalization(),
            Activation('relu'),

            Conv1D(128, 9, strides=2),
            BatchNormalization(),
            Activation('relu'),
            MaxPool1D(2, 2),

            Conv1D(256, 9, strides=1),
            BatchNormalization(),
            Activation('relu'),

            Conv1D(256, 9, strides=1),
            BatchNormalization(),
            Activation('relu'),

            Bidirectional(GRU(128)),

            Flatten(),
            Dense(4, activation='softmax')
        ])
