import os
from glob import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def make_dirs(dir):
    os.makedirs(dir, exist_ok=True)


def one_hot_encode(data, n_categories):
    if isinstance(data, list):
        for j in range(len(data)):
            data[j] = one_hot_encode(data[j], n_categories)
    else:
        data = data.astype(np.int64)
        eye = np.eye(n_categories)
        data = eye[data]
    return data


def mean_by_column(data):
    def flat_mean(d):
        new_val = pd.Series(d).apply(np.array).mean(0)
        if type(new_val) == np.ndarray:
            new_val = str(new_val)
        return new_val

    return pd.Series(data.apply(flat_mean, axis=0))


def eval_series(x):
    def fn(y):
        if type(y) == str:
            return eval(y)
        else:
            return y

    return x.apply(fn)


def predict_proba(model, dataset, from_logits=False):
    y_true = []
    y_pred = []
    for X_i, y_true_i in dataset:
        y_pred_i = model.predict(X_i)
        y_true.extend(y_true_i)
        y_pred.extend(y_pred_i)
    y_true = np.vstack(y_true).squeeze()
    y_pred = np.vstack(y_pred).squeeze()
    if from_logits:
        y_pred_proba = tf.nn.softmax(y_pred_i)
    else:
        y_pred_proba = y_pred
    return y_pred_proba, y_true


def predict_proba_for_multiple_models(models_dirs, dataset, from_logits=False):
    print(models_dirs)
    y_pred_proba = []
    y_true = []
    if type(models_dirs) == list:
        print('list')
        for p in models_dirs:
            y_pred_proba_m, y_true_m = predict_proba_for_multiple_models(p, dataset, from_logits)
            y_pred_proba.append(y_pred_proba_m)
            y_true.append(y_true_m)
    else:
        print(glob(models_dirs))
        for model_path in glob(models_dirs):
            print(model_path)
            model_m = tf.keras.models.load_model(model_path)
            y_pred_proba_m, y_true_m = predict_proba(model_m, dataset, from_logits)
            y_pred_proba.append(y_pred_proba_m)
            y_true.append(y_true_m)
    return y_pred_proba, y_true


def print_metrics(metrics):
    f_score_avg = (metrics['f_score_micro'] + metrics['f_score_macro']) / 2
    precision_avg = (metrics['precision_micro'] + metrics['precision_macro']) / 2
    recall_avg = (metrics['recall_micro'] + metrics['recall_macro']) / 2
    roc_auc_avg = (metrics['roc_auc_micro'] + metrics['roc_auc_macro']) / 2
    print('f_score={}, f_score_micro={}, f_score_macro={}, f_score_avg={}'
          .format(metrics['f_score'], metrics['f_score_micro'], metrics['f_score_macro'], f_score_avg))
    print('precision={}, precision_micro={}, precision_macro={}, precision_avg={}'
          .format(metrics['precision'], metrics['precision_micro'], metrics['precision_macro'], precision_avg))
    print('recall={}, recall_micro={}, recall_macro={}, recall_avg={}'
          .format(metrics['recall'], metrics['recall_micro'], metrics['recall_macro'], recall_avg))
    print('roc_auc={}, roc_auc_micro={}, roc_auc_macro={}, roc_auc_avg={}'
          .format(metrics['roc_auc'], metrics['roc_auc_micro'], metrics['roc_auc_macro'], roc_auc_avg))


def series_to_string(x):
    return x.apply(lambda x: np.array2string(x, separator=','))


def count_records(labels_count_path):
    labels_info = _calc_labels_info(labels_count_path)
    return labels_info['count'].sum()


def calc_class_weights(labels_count_path):
    labels_info = _calc_labels_info(labels_count_path)
    labels_info['freq'] = labels_info['count'].div(labels_info['count'].sum())
    labels_info['inv_freq'] = 1 / labels_info['freq']
    labels_info['inv_freq_norm'] = labels_info['inv_freq'] / labels_info.sum(axis=0)['inv_freq']
    return labels_info['inv_freq_norm'].to_dict()


def _calc_labels_info(labels_count_path):
    return pd.read_json(labels_count_path, typ='series') \
            .rename({'N': 0, 'A': 1, 'O': 2, '~': 3}) \
            .to_frame('count')


def get_n_classes(y_pred_probas=None, fpr=None):
    if y_pred_probas is not None:
        if isinstance(y_pred_probas, list):
            if isinstance(y_pred_probas[0], list):
                n_classes = y_pred_probas[0][0].shape[1]
            else:
                n_classes = y_pred_probas[0].shape[1]
        else:
            n_classes = y_pred_probas.shape[1]
    elif isinstance(fpr[0], list):
            n_classes = len(fpr[0])
    else:
        n_classes = len(fpr)
    return n_classes


def calc_class_weights(labels_file_path):
    labels_freq = count_labels.count_labels(labels_file_path)
    labels_freq_inv = 1/labels_freq
    labels_freq_inv_norm = labels_freq_inv/labels_freq_inv.sum()
    return labels_freq_inv_norm


def count_labels(labels_file_path, output_file_path=None):
    sample_info = pd.read_csv(labels_file_path, sep=',', header=None)
    labels = sample_info.iloc[:, 1]
    labels_count = labels.value_counts(normalize=False)
    if output_file_path is not None:
        output_dir = os.path.dirname(output_file_path)
        make_dirs(output_dir)
        labels_count.to_json(output_file_path)
    return labels_count


def calc_metrics(model=None, dataset=None, y_true=None, y_pred_probas=None):
    if y_pred_probas is None or y_true is None:
        y_pred_probas, y_true = predict_proba(model, dataset)

    n_classes = get_n_classes(y_pred_probas=y_pred_probas)

    y_true_binary = one_hot_encode(y_true, n_classes)
    y_pred = np.argmax(y_pred_probas, axis=-1)
    y_pred_binary = one_hot_encode(y_pred, n_classes)

    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    f_score = f1_score(y_true=y_true_binary, y_pred=y_pred_binary, labels=[0, 1, 2, 3], average=None)
    f_score_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f_score_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    precision = precision_score(y_true=y_true_binary, y_pred=y_pred_binary, labels=[0, 1, 2, 3], average=None)
    precision_micro = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
    precision_macro = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    recall = recall_score(y_true=y_true_binary, y_pred=y_pred_binary, labels=[0, 1, 2, 3], average=None)
    recall_micro = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
    recall_macro = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
    roc_auc = roc_auc_score(y_true=y_true_binary, y_score=y_pred_probas, average=None)
    roc_auc_micro = roc_auc_score(y_true=y_true_binary, y_score=y_pred_probas, average='micro')
    roc_auc_macro = roc_auc_score(y_true=y_true_binary, y_score=y_pred_probas, average='macro')
    all_metrics = pd.Series({
        'f_score': f_score, 'f_score_micro': f_score_micro, 'f_score_macro': f_score_macro,
        'precision': precision, 'precision_micro': precision_micro, 'precision_macro': precision_macro,
        'recall': recall, 'recall_micro': recall_micro, 'recall_macro': recall_macro,
        'roc_auc': roc_auc, 'roc_auc_micro': roc_auc_micro, 'roc_auc_macro': roc_auc_macro
    })
    val_f_score = (f_score_micro + f_score_macro) / 2
    return all_metrics, confusion_matrix, val_f_score
