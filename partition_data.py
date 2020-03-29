import argparse
import os
from glob import glob
import wfdb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow as tf

import processing

SEED = 3


def partition_data(raw_data_folder, labels_file_path, sampling_frequency, output_dir, test_set_size, n_splits,
                   baseline_wander_removal_filter=False, seed=None):
    sample_info = pd.read_csv(labels_file_path, sep=',', header=None)

    dataset = _build_dataset(raw_data_folder, sample_info, sampling_frequency,
                             baseline_wander_removal_filter=baseline_wander_removal_filter)

    X_train, X_test, y_train, y_test = train_test_split(dataset['signal'], dataset['label'], test_size=test_set_size,
                                                        shuffle=True, stratify=dataset['label'], random_state=seed)

    _write_test_set(X=X_test, y=y_test, output_dir=output_dir)

    _write_dataset_splits(X=X_train, y=y_train, output_dir=output_dir, n_splits=n_splits)


def _build_dataset(raw_data_folder, sample_info, sampling_frequency, baseline_wander_removal_filter=False):
    dataset = pd.DataFrame(columns=['signal', 'label'])
    tot_records = sample_info.shape[0]
    i = 0
    file_name_column_index, labels_column_index = 0, 1
    record_names = sorted(list(os.path.basename(f).split('.')[0] for f in glob(raw_data_folder + '/*.mat')))
    sample_info = sample_info[sample_info.iloc[:, file_name_column_index].isin(record_names)]
    for _, example_info in sample_info.iterrows():
        record_name = example_info.iloc[file_name_column_index]
        signals, _ = wfdb.rdsamp(os.path.join(raw_data_folder, record_name))
        signals = np.squeeze(signals)
        if baseline_wander_removal_filter:
            signals = processing.baseline_wander_removal(signals, sampling_frequency)
        signals = signals.tolist()
        dataset = dataset.append({'signal': signals,
                                  'label': example_info.iloc[labels_column_index]},
                                 ignore_index=True, sort=False)
        if (((i + 1) % 100) == 0) or (i + 1 == tot_records):
            print("{}/{}".format(i + 1, tot_records))
        i += 1
    return dataset


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _serialize_example(signals, class_name):
    """
  Creates a tf.Example message ready to be written to a file.
  """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'signal': _float_feature(signals),
        'label': _bytes_feature([tf.compat.as_bytes(class_name)]),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def _write_test_set(X, y, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with tf.io.TFRecordWriter(os.path.join(output_dir, "test.tf")) as writer:
        for x_i, y_i in zip(X, y):
            example = _serialize_example(x_i, y_i)
            writer.write(example)


def _write_dataset_splits(X, y, output_dir, n_splits):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for p, (_, indices) in enumerate(StratifiedKFold(n_splits=n_splits).split(X, y)):
        print("part #{}: {} indices".format(p, len(indices)))
        with tf.io.TFRecordWriter(os.path.join(output_dir, "part_{}.tf".format(p))) as writer:
            for i in indices:
                example = _serialize_example(X.values[i], y.values[i])
                writer.write(example)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-raw_data_folder", required=True)
    parser.add_argument("-sampling_frequency", type=int, required=True)
    parser.add_argument("-labels_file_path", required=True,
                        help="path to the labels csv file")
    parser.add_argument("-output_dir", required=True,
                        help="path to the output file")
    parser.add_argument("-test_set_size", type=int, required=True)
    parser.add_argument("-n_splits", type=int, required=True)
    parser.add_argument("-baseline_wander_removal_filter", type=bool, required=False, default=False)
    args = parser.parse_args()
    partition_data(raw_data_folder=args.raw_data_folder,
                   labels_file_path=args.labels_file_path,
                   output_dir=args.output_dir,
                   sampling_frequency=args.sampling_frequency,
                   test_set_size=args.test_set_size, n_splits=args.n_splits,
                   baseline_wander_removal_filter=args.baseline_wander_removal_filter, seed=SEED)
