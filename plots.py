import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
from matplotlib import pyplot as plt
from itertools import cycle
import seaborn as sn
from copy import deepcopy

import utils
import metrics


def roc_curve(y_true, y_pred_probas, class_names, thresholds=None, with_title=True, sparse=False, multi_model=False,
              model_names=None):
    if thresholds is None:
        thresholds = np.arange(0, 1.01, 0.01)
    n_classes = len(class_names)
    if sparse:
        y_true = deepcopy(y_true)
        y_true = utils.one_hot_encode(y_true, n_classes)

    # Plot all ROC curves
    fig = plt.figure()

    if multi_model:
        fpr, tpr, roc_auc = metrics.roc_curve_per_class(y_true, y_pred_probas, thresholds, multi_model=multi_model)
    else:
        fpr, tpr, roc_auc = roc_curve_per_class(y_true, y_pred_probas, class_names, thresholds, plots_combination=True)

    roc_curve_macro_average(fpr=fpr, tpr=tpr, roc_auc=roc_auc, plots_combination=True, thresholds=thresholds,
                            multi_model=multi_model, model_names=model_names)
    roc_curve_micro_average(y_true, y_pred_probas, thresholds, plots_combination=True,
                            multi_model=multi_model, model_names=model_names)

    if with_title:
        plt.title('ROC curve')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

    return fig


def roc_curve_per_class(y_true, y_pred_probas, class_names, thresholds=None, with_title=True,
                        plots_combination=False, sparse=False):
    n_classes = len(class_names)
    fpr, tpr, roc_auc = metrics.roc_curve_per_class(y_true=y_true, y_pred_probas=y_pred_probas, thresholds=thresholds,
                                                    sparse=sparse)

    if not plots_combination:
        plt.figure()

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'limegreen'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, linewidth=3,
                 label="ROC curve of '{0}' class (area = {1:0.2f})"
                       "".format(class_names[i], roc_auc[i]))

    if not plots_combination:
        if with_title:
            plt.title('ROC curve')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()

    return fpr, tpr, roc_auc


def roc_curve_average(y_true, y_pred_probas, fpr=None, tpr=None, roc_auc=None, thresholds=None,
                      with_title=True, sparse=False, multi_model=None, model_names=None):
    fpr_avg, tpr_avg, roc_auc_avg = \
        metrics.roc_curve_average(y_true=y_true, y_pred_probas=y_pred_probas, fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  thresholds=thresholds, sparse=sparse, multi_model=multi_model)

    fig = plt.figure()

    if multi_model:
        for i, (fpr_avg_i, tpr_avg_i, roc_auc_avg_i) in enumerate(zip(fpr_avg, tpr_avg, roc_auc_avg)):
            label_i = '{0} (area = {1:0.2f})'.format(model_names[i], roc_auc_avg_i)
            plt.plot(fpr_avg_i, tpr_avg_i, linewidth=3, label=label_i)
    else:
        plt.plot(fpr_avg, tpr_avg,
                 label='area = {0:0.2f}'
                       ''.format(roc_auc_avg),
                 color='deeppink', linestyle=':', linewidth=3)

    if with_title:
        plt.title('Average ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

    return fig


def roc_curve_micro_average(y_true, y_pred_probas, thresholds=None, with_title=True,
                            plots_combination=False, sparse=False, multi_model=False,
                            model_names=None):
    fpr_micro, tpr_micro, roc_auc_micro = \
        metrics.roc_curve_micro_average(y_true=y_true, y_pred_probas=y_pred_probas, thresholds=thresholds,
                                        sparse=sparse, multi_model=multi_model)

    if plots_combination:
        fig = plt.gcf()
    else:
        fig = plt.figure()

    if multi_model:
        for i, (fpr_micro_i, tpr_micro_i, roc_auc_micro_i) in enumerate(zip(fpr_micro, tpr_micro, roc_auc_micro)):
            label_i = '{0} (area = {1:0.2f})'.format(model_names[i], roc_auc_micro_i)
            if plots_combination:
                label_i = 'micro-average ROC curve of ' + label_i
            plt.plot(fpr_micro_i, tpr_micro_i, linewidth=3, label=label_i)
    else:
        label_i = '(area = {0:0.2f})'.format(roc_auc_micro)
        if plots_combination:
            label_i = 'micro-average ROC curve ' + label_i
        plt.plot(fpr_micro, tpr_micro,
                 label=label_i,
                 color='deeppink', linestyle=':', linewidth=3)

    if not plots_combination:
        if with_title:
            plt.title('Micro-average ROC curve')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()

    return fig


def roc_curve_macro_average(y_true=None, y_pred_probas=None, fpr=None, tpr=None, roc_auc=None,
                            thresholds=None, with_title=True, plots_combination=False, sparse=False, multi_model=False,
                            model_names=None):
    fpr_macro, tpr_macro, roc_auc_macro = metrics.roc_curve_macro_average(y_true=y_true, y_pred_probas=y_pred_probas,
                                                                          fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                                                          thresholds=thresholds, sparse=sparse,
                                                                          multi_model=multi_model)

    if plots_combination:
        fig = plt.gcf()
    else:
        fig = plt.figure()

    if multi_model:
        for i, (fpr_macro_i, tpr_macro_i, roc_auc_macro_i) in enumerate(zip(fpr_macro, tpr_macro, roc_auc_macro)):
            label_i = '{0} (area = {1:0.2f})'.format(model_names[i], roc_auc_macro_i)
            if plots_combination:
                label_i = 'macro-average ROC curve of ' + label_i
            plt.plot(fpr_macro_i, tpr_macro_i, linewidth=3, label=label_i)
    else:
        label_i = '(area = {0:0.2f})'.format(roc_auc_macro)
        if plots_combination:
            label_i = 'macro-average ROC curve ' + label_i
        plt.plot(fpr_macro, tpr_macro, label=label_i, color='navy', linestyle=':', linewidth=3)

    if not plots_combination:
        if with_title:
            plt.title('Macro-average ROC curve')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()

    return fig


def precision_recall_curve(y_true, y_pred_probas, class_names, thresholds=None, with_title=True, sparse=False):
    if thresholds is None:
        thresholds = np.arange(0, 1.01, 0.01)
    n_classes = len(class_names)
    if sparse:
        y_true = deepcopy(y_true)
        y_true = utils.one_hot_encode(y_true, n_classes)

    fig = plt.figure(figsize=(7, 8))

    precision_recall_curve_per_class(y_true, y_pred_probas, class_names, thresholds=thresholds, plots_combination=True)
    precision_recall_curve_micro_average(y_true, y_pred_probas, thresholds=thresholds, plots_combination=True)

    # fig = plt.gcf()
    # fig.subplots_adjust(bottom=0.25)
    _iso_f1_curves()
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")

    if with_title:
        plt.title('Precision-recall curve')

    plt.show()
    return fig


def precision_recall_curve_per_class(y_true, y_pred_probas, class_names, thresholds=None, sparse=False, with_title=True,
                                     plots_combination=False):
    precision, recall, average_precision = \
        metrics.precision_recall_curve_per_class(y_true, y_pred_probas, thresholds=thresholds, sparse=sparse)

    if plots_combination:
        fig = plt.gcf()
    else:
        fig = plt.figure()

    n_classes = len(class_names)
    for c in range(n_classes):
        label = '{0} (area = {1:0.2f})'.format(class_names[c], average_precision[c])
        if plots_combination:
            label = 'Precision-recall for ' + label
        plt.plot(recall[c], precision[c], linewidth=3, label=label)

    if not plots_combination:
        if with_title:
            plt.title('Precision-recall curve per class')
        _iso_f1_curves()
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="lower left")
        plt.show()
    return fig


def precision_recall_curve_micro_average(y_true, y_pred_probas, thresholds=None, sparse=False,
                                         with_title=True, plots_combination=False, multi_model=False,
                                         model_names=None):
    precision_micro, recall_micro, average_precision_micro = \
        metrics.precision_recall_curve_micro_average(y_true, y_pred_probas, thresholds=thresholds, sparse=sparse,
                                                     multi_model=multi_model)

    if plots_combination:
        fig = plt.gcf()
    else:
        fig = plt.figure()

    if multi_model:
        for i, (precision_micro_i, recall_micro_i, average_precision_micro_i) \
                in enumerate(zip(precision_micro, recall_micro, average_precision_micro)):
            label = '{0} (area = {1:0.2f})'.format(model_names[i], average_precision_micro_i)
            if plots_combination:
                label = 'micro-average Precision-recall of ' + label
            plt.plot(precision_micro_i, recall_micro_i, linewidth=3, label=label)
    else:
        label = '(area = {:0.2f})'.format(average_precision_micro)
        if plots_combination:
            label = 'micro-average Precision-recall ' + label
        plt.plot(precision_micro, recall_micro, linestyle=':', linewidth=3, label=label)

    if not plots_combination:
        if with_title:
            plt.title('Micro-average precision-recall curve per class')
        _iso_f1_curves()
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="lower left")
        plt.show()
    return fig


def _iso_f1_curves():
    f_scores = np.linspace(0.2, 0.8, num=4)

    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        curve, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.95, y[45] + 0.02))
    curve.set_label('iso-f1 curves'.format(f_score))


def confusion_matrix(y_true, y_pred_probas, class_names, with_title=True):
    n_classes = len(class_names)
    if not isinstance(y_true, list):
        y_true = [y_true]
        y_pred_probas = [y_pred_probas]
    _confusion_matrix = np.zeros((n_classes, n_classes))
    for y_true_i, y_pred_probas_i in zip(y_true, y_pred_probas):
        y_pred_i = np.argmax(y_pred_probas_i, axis=-1)
        _confusion_matrix += sklearn.metrics.confusion_matrix(y_true_i, y_pred_i, labels=[0, 1, 2, 3])
    n_blocks = len(y_true)
    _confusion_matrix /= n_blocks
    df_cm = pd.DataFrame(_confusion_matrix, index=class_names, columns=class_names) \
        .rename_axis('Predicted label', axis=1) \
        .rename_axis('True label', axis=0)
    fig = plt.figure(figsize=(8, 7))
    plt.rc('font', size=20)  # controls default text sizes
    plt.rc('axes', titlesize=20)  # fontsize of the axes title
    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=20)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=20)  # fontsize of the tick labels
    plt.rc('legend', fontsize=20)  # legend fontsize
    plt.rc('figure', titlesize=20)  # fontsize of the figure title
    sn.heatmap(df_cm, fmt=".1f", annot=True, annot_kws={"size": 20, "weight": 'bold'}, cbar=False)
    if with_title:
        plt.title('Confusion matrix')
    fig.show()
    return fig
