import tensorflow as tf
from processing import normalize, pad, crop


class AFDatasetBuilder:
    def __init__(self, target_record_len):
        self._target_record_len = target_record_len

    features_types = {
        'signal': tf.io.VarLenFeature(tf.float32),
        'label': tf.io.FixedLenFeature(shape=1, dtype=tf.string, default_value='')
    }

    labels_indices = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(keys=['N', 'A', 'O', '~'], values=[0, 1, 2, 3], value_dtype=tf.int64),
        num_oov_buckets=4)

    @classmethod
    def parse_example(cls, example, target_len, training):
        features = tf.io.parse_single_example(example, cls.features_types)
        features['signal'] = features['signal'].values
        features['signal'] = normalize(features['signal'])
        features['signal'] = pad(features['signal'], target_len)
        features['signal'] = crop(features['signal'], target_len, training)
        features['signal'] = tf.reshape(features['signal'], [-1, 1])
        features['label'] = cls.labels_indices.lookup(features['label'])
        features['label'] = tf.squeeze(features['label'])
        return features['signal'], features['label']

    def get_dataset(self, filepath, target_record_len, n_records=None, batch_size=None, training=True, seed=None):
        data = tf.data.TFRecordDataset(filepath)

        if training and n_records is not None:
            data = data.shuffle(n_records, seed=seed)
        data = data.map(lambda d: self.parse_example(d, target_len=target_record_len, training=training))
        if batch_size is not None:
            data = data.batch(batch_size)
        return data.prefetch(tf.data.experimental.AUTOTUNE)

    def get_training_set(self, paths, n_records, batch_size, seed=None):
        return self.get_dataset(paths, self._target_record_len, n_records=n_records, batch_size=batch_size,
                                training=True, seed=seed)

    def get_validation_set(self, paths, batch_size, seed=None):
        return self.get_dataset(paths, self._target_record_len, batch_size=batch_size, training=False, seed=seed)

    def get_test_set(self, paths, batch_size, seed=None):
        return self.get_dataset(paths, self._target_record_len, batch_size=batch_size, training=False, seed=seed)
