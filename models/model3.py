import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, BatchNormalization, Activation, Flatten, Dropout, \
                                    SeparableConv1D, AvgPool1D


class Model3(tf.keras.Sequential):
    def __init__(self, record_size, seed=None):
        super().__init__([
            Input(shape=(record_size, 1)),

            Conv1D(filters=32, kernel_size=16, strides=2),
            BatchNormalization(),
            Activation('relu'),

            SeparableConv1D(filters=64, kernel_size=16, strides=1, padding='SAME'),
            BatchNormalization(),
            Activation('relu'),

            SeparableConv1D(filters=128, kernel_size=16, strides=2),
            BatchNormalization(),
            Activation('relu'),

            SeparableConv1D(filters=128, kernel_size=16, strides=1, padding='SAME'),
            BatchNormalization(),
            Activation('relu'),

            SeparableConv1D(filters=256, kernel_size=16, strides=2),
            BatchNormalization(),
            Activation('relu'),

            SeparableConv1D(filters=256, kernel_size=16, strides=1, padding='SAME'),
            BatchNormalization(),
            Activation('relu'),

            SeparableConv1D(filters=512, kernel_size=16, strides=2),
            BatchNormalization(),
            Activation('relu'),

            SeparableConv1D(filters=512, kernel_size=16, strides=1, padding='SAME'),
            BatchNormalization(),
            Activation('relu'),

            SeparableConv1D(filters=512, kernel_size=16, strides=2),
            BatchNormalization(),
            Activation('relu'),

            SeparableConv1D(filters=512, kernel_size=16, strides=1, padding='SAME'),
            BatchNormalization(),
            Activation('relu'),

            SeparableConv1D(filters=512, kernel_size=16, strides=2),
            BatchNormalization(),
            Activation('relu'),

            SeparableConv1D(filters=512, kernel_size=16, strides=1, padding='SAME'),
            BatchNormalization(),
            Activation('relu'),

            SeparableConv1D(filters=1024, kernel_size=16, strides=2),
            BatchNormalization(),
            Activation('relu'),

            SeparableConv1D(filters=1024, kernel_size=16, strides=1, padding='SAME'),
            BatchNormalization(),
            Activation('relu'),

            AvgPool1D(),

            Flatten(),

            Dropout(0.3, seed=seed),
            Dense(4, activation='softmax')
        ])
