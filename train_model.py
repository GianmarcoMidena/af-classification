import argparse
import os

import models
import utils
from af_dataset_splitter import AFDatasetSplitter
from learner import Learner

SEED = 3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", required=True)
    parser.add_argument("-test_set_path", required=True)
    parser.add_argument("-model_name", required=True)
    parser.add_argument("-n_folds", type=int, required=True,
                        help="number of folds for cross validation")
    parser.add_argument("-max_n_epochs", type=int, required=True)
    parser.add_argument("-initial_lr", type=float, required=True)
    parser.add_argument("-lr_reduction_factor", type=float, required=True)
    parser.add_argument("-early_stopping_patience", type=int, required=True)
    parser.add_argument("-lr_reducer_patience", type=int, required=True)
    parser.add_argument("-batch_size", type=int, required=True)
    parser.add_argument("-record_size", type=int, required=True)
    parser.add_argument("-n_records", type=int, required=True)
    parser.add_argument("-n_test_set_records", type=int, required=True)
    parser.add_argument("-fixed_fold", type=int, required=False)
    parser.add_argument("-weights_file_path", required=False)
    parser.add_argument("-output_model_path", required=False)
    parser.add_argument("-output_metrics_path", required=False)
    parser.add_argument("-labels_file_path", required=False, help="path to the labels csv file")
    parser.add_argument("-class_weighting", type=bool, required=False, default=False)
    parser.add_argument("-static_lr", required=False, type=bool, default=False)
    args = parser.parse_args()

    model_name = args.model_name
    record_size = args.record_size
    if model_name == 'model1':
        model = models.Model1(record_size=record_size)
    elif model_name == 'model2':
        model = models.Model2(record_size=record_size)
    elif model_name == 'model3':
        model = models.Model3(record_size=record_size, seed=SEED)
    else:
        raise Exception(f"Attention: '{model_name}' does not exist!")
    data_dir = args.data_dir
    full_train_set_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if 'part' in file]
    splitter = AFDatasetSplitter(n_splits=args.n_folds, full_train_set_paths=full_train_set_paths,
                                 test_set_paths=args.test_set_path, batch_size=args.batch_size,
                                 n_records=args.n_records, n_test_set_records=args.n_test_set_records,
                                 target_record_len=record_size, seed=SEED)

    if args.class_weighting:
        class_weights = utils.calc_class_weights(args.labels_file_path)
    else:
        class_weights = None

    Learner(model, splitter, lr=args.initial_lr, lr_reduction_factor=args.lr_reduction_factor,
            early_stopping_patience=args.early_stopping_patience, lr_reducer_patience=args.lr_reducer_patience,
            static_lr=args.static_lr, class_weights=class_weights, max_n_epochs=args.max_n_epochs,
            weights_file_path=args.weights_file_path, fixed_fold=args.fixed_fold,
            output_model_path=args.output_model_path, output_metrics_path=args.output_metrics_path).fit()
