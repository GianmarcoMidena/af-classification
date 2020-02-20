import numpy as np
import sklearn
import utils
from copy import deepcopy


def true_positive_rate(y_true, y_pred_proba, thresholds):
    tprs = []
    for threshold in thresholds:
        y_pred = 1 * (y_pred_proba >= threshold)
        tp = (1 * ((y_true + y_pred) == 2)).sum()
        fn = (1 * ((y_true - y_pred) == 1)).sum()
        tprs.append(tp / (tp + fn))
    return np.asarray(tprs)


def false_positive_rate(y_true, y_pred_proba, thresholds):
    fprs = []
    for threshold in thresholds:
        y_pred = 1 * (y_pred_proba >= threshold)
        fp = (1 * ((y_pred - y_true) == 1)).sum()
        tn = (1 * ((y_true + y_pred) == 0)).sum()
        fprs.append(fp / (fp + tn))
    return np.asarray(fprs)


recall = true_positive_rate


def precision(y_true, y_pred_proba, thresholds):
    precs = []
    for threshold in thresholds:
        y_pred = 1 * (y_pred_proba >= threshold)
        tp = (1 * ((y_true + y_pred) == 2)).sum()
        fp = (1 * ((y_pred - y_true) == 1)).sum()
        precs.append(tp / (tp + fp))
    return np.asarray(precs)


def roc_curve_per_class(y_true, y_pred_probas, thresholds=None, sparse=False, multi_model=False):
    # Compute ROC curve and ROC area for each class

    if thresholds is None:
        thresholds = np.arange(0, 1.01, 0.01)

    n_classes = utils.get_n_classes(y_pred_probas)

    if sparse:
        y_true = deepcopy(y_true)
        y_true = utils.one_hot_encode(y_true, n_classes)

    if multi_model:
        n_models = len(y_true)
        fpr = [0] * n_models
        tpr = [0] * n_models
        roc_auc = [0] * n_models
        for i in range(n_models):
            fpr[i], tpr[i], roc_auc[i] = roc_curve_per_class(y_true[i], y_pred_probas[i], thresholds)
        return fpr, tpr, roc_auc

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        if isinstance(y_true, list):
            fpr[i] = None
            tpr[i] = None
            n_blocks = len(y_true)
            for j in range(n_blocks):
                fpr_j = false_positive_rate(y_true[j][:, i], y_pred_probas[j][:, i], thresholds)
                tpr_j = true_positive_rate(y_true[j][:, i], y_pred_probas[j][:, i], thresholds)
                if fpr[i] is None:
                    fpr[i] = fpr_j
                    tpr[i] = tpr_j
                else:
                    fpr[i] += fpr_j
                    tpr[i] += tpr_j
            fpr[i] /= n_blocks
            tpr[i] /= n_blocks
        else:
            fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_true=y_true[:, i],
                                                          y_score=y_pred_probas[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc


def roc_curve_average(y_true, y_pred_probas, fpr=None, tpr=None, roc_auc=None, thresholds=None,
                      sparse=False, multi_model=False):
    if thresholds is None:
        thresholds = np.arange(0, 1.01, 0.01)

    n_classes = utils.get_n_classes(y_pred_probas)

    if sparse:
        y_true = deepcopy(y_true)
        y_true = utils.one_hot_encode(y_true, n_classes)

    fpr_macro, tpr_macro, roc_auc_macro = \
        roc_curve_macro_average(y_true=y_true, y_pred_probas=y_pred_probas, fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                thresholds=thresholds, multi_model=multi_model)

    fpr_micro, tpr_micro, roc_auc_micro = \
        roc_curve_micro_average(y_true, y_pred_probas, thresholds=thresholds, multi_model=multi_model)

    if isinstance(fpr_micro, list):
        n_models = len(fpr_micro)
        fpr_average = [0] * n_models
        tpr_average = [0] * n_models
        roc_auc_average = [0] * n_models
        for i in range(n_models):
            fpr_average[i] = (fpr_micro[i] + fpr_macro[i]) / 2
            tpr_average[i] = (tpr_micro[i] + tpr_macro[i]) / 2
            roc_auc_average[i] = (roc_auc_micro[i] + roc_auc_macro[i]) / 2
    else:
        fpr_average = (fpr_micro + fpr_macro) / 2
        tpr_average = (tpr_micro + tpr_macro) / 2
        roc_auc_average = (roc_auc_micro + roc_auc_macro) / 2
    return fpr_average, tpr_average, roc_auc_average


def roc_curve_micro_average(y_true, y_pred_probas, thresholds=None, sparse=False, multi_model=False):
    if thresholds is None:
        thresholds = np.arange(0, 1.01, 0.01)

    n_classes = utils.get_n_classes(y_pred_probas)

    if sparse:
        y_true = deepcopy(y_true)
        y_true = utils.one_hot_encode(y_true, n_classes)

    # Compute micro-average ROC curve and ROC area
    if multi_model:
        fpr_micro = []
        tpr_micro = []
        roc_auc_micro = []
        for y_true_i, y_pred_probas_i in zip(y_true, y_pred_probas):
            fpr_micro_i, tpr_micro_i, roc_auc_micro_i = _roc_curve_micro_average_single_model(y_true_i, y_pred_probas_i,
                                                                                              thresholds)
            fpr_micro.append(fpr_micro_i)
            tpr_micro.append(tpr_micro_i)
            roc_auc_micro.append(roc_auc_micro_i)
    else:
        fpr_micro, tpr_micro, roc_auc_micro = _roc_curve_micro_average_single_model(y_true, y_pred_probas, thresholds)
    return fpr_micro, tpr_micro, roc_auc_micro


def _roc_curve_micro_average_single_model(y_true, y_pred_probas, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0, 1.01, 0.01)

    # Compute micro-average ROC curve and ROC area
    if isinstance(y_true, list):
        fpr_micro = None
        tpr_micro = None
        n_blocks = len(y_true)
        for j in range(n_blocks):
            fpr_j = false_positive_rate(y_true[j].ravel(), y_pred_probas[j].ravel(), thresholds)
            tpr_j = true_positive_rate(y_true[j].ravel(), y_pred_probas[j].ravel(), thresholds)
            if fpr_micro is None:
                fpr_micro = fpr_j
                tpr_micro = tpr_j
            else:
                fpr_micro += fpr_j
                tpr_micro += tpr_j
        fpr_micro /= n_blocks
        tpr_micro /= n_blocks
    else:
        fpr_micro, tpr_micro, _ = sklearn.metrics.roc_curve(y_true.ravel(), y_pred_probas.ravel())
    roc_auc_micro = sklearn.metrics.auc(fpr_micro, tpr_micro)
    return fpr_micro, tpr_micro, roc_auc_micro


def roc_curve_macro_average(y_true=None, y_pred_probas=None, fpr=None, tpr=None, roc_auc=None,
                            thresholds=None, sparse=False, multi_model=False):
    # Compute micro-average ROC curve and ROC area

    n_classes = utils.get_n_classes(y_pred_probas=y_pred_probas, fpr=fpr)

    if fpr is None or tpr is None or roc_auc is None:
        if thresholds is None:
            thresholds = np.arange(0, 1.01, 0.01)

        if sparse and y_true:
            y_true = deepcopy(y_true)
            y_true = utils.one_hot_encode(y_true, n_classes)

    if multi_model:
        if y_true is None:
            n_models = len(fpr)
        else:
            n_models = len(y_true)
        fpr_macro = [0] * n_models
        tpr_macro = [0] * n_models
        roc_auc_macro = [0] * n_models
        if y_true is None:
            for i in range(n_models):
                fpr_macro[i], tpr_macro[i], roc_auc_macro[i] = \
                    roc_curve_macro_average(fpr=fpr[i], tpr=tpr[i], roc_auc=roc_auc[i], thresholds=thresholds,
                                            sparse=False, multi_model=False)
        else:
            for i in range(n_models):
                fpr_macro[i], tpr_macro[i], roc_auc_macro[i] = \
                    roc_curve_macro_average(y_true=y_true[i], y_pred_probas=y_pred_probas[i], thresholds=thresholds,
                                            sparse=False, multi_model=False)
        return fpr_macro, tpr_macro, roc_auc_macro

    if fpr is None or tpr is None or roc_auc is None:
        fpr, tpr, roc_auc = roc_curve_per_class(y_true, y_pred_probas, thresholds=thresholds, sparse=sparse)

    fpr_macro = fpr[0]
    tpr_macro = tpr[0]
    roc_auc_macro = roc_auc[0]
    for i in range(1, n_classes):
        fpr_macro += fpr[i]
        tpr_macro += tpr[i]
        roc_auc_macro += roc_auc[i]
    fpr_macro /= n_classes
    tpr_macro /= n_classes
    roc_auc_macro /= n_classes
    return fpr_macro, tpr_macro, roc_auc_macro


def precision_recall_curve_per_class(y_true, y_pred_probas, thresholds=None, sparse=False):
    if thresholds is None:
        thresholds = np.arange(0, 1.01, 0.01)

    n_classes = utils.get_n_classes(y_pred_probas)

    if sparse:
        y_true = deepcopy(y_true)
        y_true = utils.one_hot_encode(y_true, n_classes)

    _precision = dict()
    _recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        if isinstance(y_true, list):
            _precision[i] = None
            _recall[i] = None
            n_blocks = len(y_true)
            average_precision[i] = None
            for j in range(n_blocks):
                precision_j = precision(y_true[j][:, i], y_pred_probas[j][:, i], thresholds)
                recall_j = recall(y_true[j][:, i], y_pred_probas[j][:, i], thresholds)
                average_precision_j = sklearn.metrics.average_precision_score(y_true[j][:, i], y_pred_probas[j][:, i])
                if _precision[i] is None:
                    _precision[i] = precision_j
                    _recall[i] = recall_j
                    average_precision[i] = average_precision_j
                else:
                    _precision[i] += precision_j
                    _recall[i] += recall_j
                    average_precision[i] += average_precision_j
            _precision[i] /= n_blocks
            _recall[i] /= n_blocks
            average_precision[i] /= n_blocks
        else:
            _precision[i], _recall[i], _ = sklearn.metrics.precision_recall_curve(y_true[:, i],
                                                                                  y_pred_probas[:, i])
            average_precision[i] = sklearn.metrics.average_precision_score(y_true[:, i], y_pred_probas[:, i])
    return _precision, _recall, average_precision


def precision_recall_curve_micro_average(y_true, y_pred_probas, thresholds=None, sparse=False, multi_model=False):
    # A "micro-average": quantifying score on all classes jointly

    if thresholds is None:
        thresholds = np.arange(0, 1.01, 0.01)

    n_classes = utils.get_n_classes(y_pred_probas)

    if sparse:
        y_true = deepcopy(y_true)
        y_true = utils.one_hot_encode(y_true, n_classes)

    if multi_model:
        n_models = len(y_true)
        precision_micro = [0] * n_models
        recall_micro = [0] * n_models
        average_precision_micro = [0] * n_models
        for j in range(n_models):
            precision_micro[j], recall_micro[j], average_precision_micro[j] = \
                precision_recall_curve_micro_average(y_true[j], y_pred_probas[j], thresholds=thresholds, sparse=False)
        return precision_micro, recall_micro, average_precision_micro

    if isinstance(y_true, list):
        precision_micro = None
        recall_micro = None
        average_precision_micro = None
        n_blocks = len(y_true)
        for j in range(n_blocks):
            precision_j = precision(y_true[j].ravel(), y_pred_probas[j].ravel(), thresholds)
            recall_j = recall(y_true[j].ravel(), y_pred_probas[j].ravel(), thresholds)
            average_precision_j = sklearn.metrics.average_precision_score(y_true[j], y_pred_probas[j], average="micro")
            if precision_micro is None:
                precision_micro = precision_j
                recall_micro = recall_j
                average_precision_micro = average_precision_j
            else:
                precision_micro += precision_j
                recall_micro += recall_j
                average_precision_micro += average_precision_j
        precision_micro /= n_blocks
        recall_micro /= n_blocks
        average_precision_micro /= n_blocks
    else:
        precision_micro, recall_micro, _ = sklearn.metrics.precision_recall_curve(y_true.ravel(), y_pred_probas.ravel())
        average_precision_micro = sklearn.metrics.average_precision_score(y_true, y_pred_probas, average="micro")
    return precision_micro, recall_micro, average_precision_micro
