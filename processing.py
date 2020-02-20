import numpy as np
import scipy as sp
import tensorflow as tf
import tensorflow_probability as tfp


def baseline_wander_removal(data, sampling_frequency):
    # Baseline estimation
    win_size = int(np.round(0.2 * sampling_frequency)) + 1
    baseline = sp.signal.medfilt(data, win_size)
    win_size = int(np.round(0.6 * sampling_frequency)) + 1
    baseline = sp.signal.medfilt(baseline, win_size)
    # Removing baseline
    filt_data = data - baseline
    return filt_data


def normalize(data):
    _5th_percentile = tfp.stats.percentile(data, 5)
    _95th_percentile = tfp.stats.percentile(data, 95)
    return data / (_95th_percentile - _5th_percentile)


def pad(data, target_len):
    original_len = tf.shape(data)[0]
    padding_len = target_len - original_len

    def _pad(d):
        d = tf.expand_dims(d, -1)
        left_padding_len = tf.math.ceil(padding_len / 2)
        right_padding_len = tf.math.floor(padding_len / 2)
        top_padding_len = 0
        down_padding_len = 0
        paddings = [[left_padding_len, right_padding_len], [top_padding_len, down_padding_len]]
        return tf.pad(d, paddings=paddings)

    return tf.cond(tf.greater(padding_len, 0),
                   true_fn=lambda: _pad(data), false_fn=lambda: data)


def crop(data, target_len, training=False, seed=None):
    original_len = tf.shape(data)[0]

    def _crop(_data, _training):
        if _training:
            _from = tf.random.uniform(shape=(1,), minval=0,
                                      maxval=original_len - target_len,
                                      dtype=tf.int32, seed=seed)
            _from = tf.squeeze(_from)
        else:  # center crop
            _from = tf.subtract(tf.cast(tf.floor(original_len / 2), tf.int32),
                                tf.cast(tf.floor(target_len / 2), tf.int32))
        _to = _from + target_len
        return _data[_from:_to]

    return tf.cond(tf.greater(original_len, target_len),
                   true_fn=lambda: _crop(data, training), false_fn=lambda: data)
