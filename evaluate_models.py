import argparse
import os
import pandas as pd

from af_dataset_builder import AFDatasetBuilder
import plots
import utils

CLASS_NAMES = ['normal', 'af', 'other', 'noise']

SEED = 3


def summarize_metrics(models_path, test_set_path, target_record_len, batch_size, model_name=None, output_path=None):
    all_metrics = pd.DataFrame()

    test_set = AFDatasetBuilder(target_record_len).get_test_set(test_set_path, batch_size=batch_size, seed=SEED)
    y_test_pred_proba, y_test = utils.predict_proba_for_multiple_models(models_path, test_set)

    for y_test_pred_proba_i, y_test_i in zip(y_test_pred_proba, y_test):
        metrics_i, _, _ = utils.calc_metrics(y_true=y_test_i, y_pred_probas=y_test_pred_proba_i)
        all_metrics = all_metrics.append(metrics_i.to_frame().transpose(), ignore_index=True, sort=False) \
            .reset_index(drop=True)

    metrics_summary = utils.mean_by_column(all_metrics)
    if model_name is not None:
        print(model_name)
    utils.print_metrics(metrics_summary)

    if output_path is not None:
        output_dir = os.path.split(output_path)[0]
        utils.make_dirs(output_dir)
        metrics_summary.to_frame().transpose().to_csv(output_path, index=False)


def plot_model(models_path, test_set_path, target_record_len, batch_size,
               class_names=CLASS_NAMES, model_name=None, output_plot_dir=None):
    test_set = AFDatasetBuilder(target_record_len).get_test_set(test_set_path, batch_size=batch_size, seed=SEED)
    y_test_pred_proba, y_test = utils.predict_proba_for_multiple_models(models_path, test_set)

    plots_collection = {
        'confusion_matrix': plots.confusion_matrix(y_true=y_test, y_pred_probas=y_test_pred_proba,
                                                   class_names=class_names, with_title=False),
        'roc_curve': plots.roc_curve(y_true=y_test, y_pred_probas=y_test_pred_proba,
                                     class_names=class_names, sparse=True, with_title=False),
        'pr_curve': plots.precision_recall_curve(y_true=y_test, y_pred_probas=y_test_pred_proba,
                                                 class_names=class_names, sparse=True, with_title=False)
    }

    if output_plot_dir is not None:
        if model_name is None:
            raise Exception("A model name is required for saving plots")
        else:
            utils.make_dirs(output_plot_dir)
            for name, fig in plots_collection.items():
                if model_name:
                    filename = f'{name}_{model_name}.pdf'
                if output_plot_dir:
                    filepath = os.path.join(output_plot_dir, filename)
                else:
                    filepath = filename
                fig.savefig(filepath)


def plot_models(models_paths, model_names, test_set_path, target_record_len, batch_size, output_plot_dir=None,
                class_names=CLASS_NAMES):
    test_set = AFDatasetBuilder(target_record_len).get_test_set(test_set_path, batch_size=batch_size, seed=SEED)
    y_test_pred_proba, y_test = utils.predict_proba_for_multiple_models(models_paths, test_set)

    plots_collection = {
        'roc_curve': plots.roc_curve(y_true=y_test, y_pred_probas=y_test_pred_proba,
                                     class_names=class_names, multi_model=True, model_names=model_names,
                                     with_title=False, sparse=True),
        'roc_curve_average': plots.roc_curve_average(y_true=y_test, y_pred_probas=y_test_pred_proba,
                                                     multi_model=True, model_names=model_names, with_title=False,
                                                     sparse=True),
        'roc_curve_micro_average': plots.roc_curve_micro_average(y_true=y_test, y_pred_probas=y_test_pred_proba,
                                                                 multi_model=True, model_names=model_names,
                                                                 with_title=False, sparse=True),
        'roc_curve_macro_average': plots.roc_curve_macro_average(y_true=y_test, y_pred_probas=y_test_pred_proba,
                                                                 multi_model=True, model_names=model_names,
                                                                 with_title=False, sparse=True),
        'precision_recall_curve_micro_average':
            plots.precision_recall_curve_micro_average(y_true=y_test, y_pred_probas=y_test_pred_proba, sparse=True,
                                                       multi_model=True, model_names=model_names, with_title=False)
    }
    if output_plot_dir is not None:
        utils.make_dirs(output_plot_dir)
        for name, fig in plots_collection.items():
            filename = f'{name}.pdf'
            if output_plot_dir:
                filepath = os.path.join(output_plot_dir, filename)
            else:
                filepath = filename
            fig.savefig(filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-test_set_path", required=True)
    parser.add_argument("-record_size", type=int, required=True)
    parser.add_argument("-batch_size", type=int, required=True)
    parser.add_argument("-models_paths", nargs='+', required=False)
    parser.add_argument("-model_names", nargs='+', required=False)
    parser.add_argument("-output_plot_dir", required=False)
    args = parser.parse_args()

    models_paths = args.models_paths
    test_set_path = args.test_set_path
    record_size = args.record_size
    batch_size = args.batch_size
    output_plot_dir = args.output_plot_dir
    model_names = args.model_names
    if models_paths is not None:
        for i, model_path in enumerate(models_paths):
            if model_names is not None:
                model_name = args.model_names[i]
            else:
                model_name = None
            summarize_metrics(models_path=model_path, test_set_path=test_set_path, target_record_len=record_size,
                              batch_size=batch_size, model_name=model_name, output_path=output_plot_dir)
            plot_model(models_path=model_path, test_set_path=test_set_path, target_record_len=record_size,
                       batch_size=batch_size, model_name=model_name, output_plot_dir=output_plot_dir)

    if models_paths is not None and len(models_paths) > 1:
        plot_models(models_paths=models_paths, model_names=model_names, test_set_path=test_set_path,
                    target_record_len=record_size, batch_size=batch_size, output_plot_dir=output_plot_dir)
