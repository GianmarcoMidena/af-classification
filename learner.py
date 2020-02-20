import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy

import utils


class Learner:
    def __init__(self, model, splitter, lr, lr_reduction_factor=0.1, early_stopping_patience=15, lr_reducer_patience=5,
                 static_lr=False, class_weights=None, loss_fn=SparseCategoricalCrossentropy, optimizer=Adam,
                 max_n_epochs=200, weights_file_path=None, fixed_fold=None, output_model_path=None,
                 output_metrics_path=None):
        self._model = tf.keras.models.clone_model(model)
        self._split = splitter.split(fixed_fold=fixed_fold)
        self._lr = lr
        self._lr_reduction_factor = lr_reduction_factor
        self._lr_reducer_patience = lr_reducer_patience
        self._static_lr = static_lr
        self._class_weights = class_weights
        self._early_stopping_patience = early_stopping_patience
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._max_n_epochs = max_n_epochs
        self._weights_file_path = weights_file_path
        self._fixed_fold = fixed_fold
        self._output_model_path = output_model_path
        self._output_metrics_path = output_metrics_path
        self._reset_state()

    def fit(self):
        all_metrics = pd.DataFrame()
        fold_counter = 0

        for training_set, validation_set, test_set in self._split:
            fold_counter += 1
            if self._fixed_fold is None:
                print(f'\nfold #{fold_counter}')
            else:
                print(f'\nfold #{self._fixed_fold}')

            model = self._build_model()

            self._reset_state()
            for epoch in range(self._max_n_epochs):
                val_loss = self._fit(model, training_set, validation_set, epoch)

                val_metrics, val_confusion_matrix, val_f_score = utils.calc_metrics(model=model, dataset=validation_set)

                self._update_state(model, val_loss, val_f_score, epoch)

                if ((epoch + 1) % 5) == 0 or (self._lr_age == 0) or self._is_early_stopping() or \
                   (epoch + 1) == self._max_n_epochs:
                    print("\nValidation:")
                    print(val_confusion_matrix)
                    utils.print_metrics(val_metrics)
                    print("\n")
                else:
                    print(f"val_f_score={val_f_score:.4f}")

                if self._is_early_stopping():
                    break

            if self._output_model_path is not None:
                self._best_model.save(self._output_model_path)

            print(f"\nbest_epoch={self._best_epoch}")
            print("\nValidation:")
            val_metrics, val_confusion_matrix, _ = utils.calc_metrics(model=self._best_model, dataset=validation_set)
            print(val_confusion_matrix)
            utils.print_metrics(val_metrics)

            print("\nTesting:")
            new_metrics, confusion_matrix, _ = utils.calc_metrics(model=self._best_model, dataset=test_set)
            print(confusion_matrix)
            utils.print_metrics(new_metrics)
            all_metrics = all_metrics.append(new_metrics.to_frame().transpose(), ignore_index=True, sort=False) \
                .reset_index(drop=True)
            print("\n")

        if self._output_metrics_path is not None:
            output_dir = os.path.split(self._output_metrics_path)[0]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            all_metrics.apply(utils.series_to_string).to_csv(self._output_metrics_path, index=False)

        return all_metrics

    def _build_model(self):
        model = tf.keras.models.clone_model(self._model)

        if self._weights_file_path is not None:
            model.load_weights(self._weights_file_path)

        model.compile(optimizer=self._optimizer(self._lr), loss=self._loss_fn(),
                      metrics=[SparseCategoricalAccuracy(name="acc")])
        return model

    def _reset_state(self):
        self._reset_learning_rate_age()
        self._best_val_loss_age = 0
        self._best_val_loss = float('inf')
        self._best_f_score = 0
        self._best_epoch = None
        self._best_model = None

    def _update_state(self, model, val_loss, val_f_score, epoch):
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._best_val_loss_age = 0
        else:
            self._best_val_loss_age += 1

        if val_f_score > self._best_f_score:
            self._best_f_score = val_f_score
            self._best_epoch = epoch + 1
            self._best_model = tf.keras.models.clone_model(model)

        self._update_learning_rate(model)

    def _update_learning_rate(self, model):
        if self._is_learning_rate_to_reduce():
            self._reduce_learning_rate(model)
        else:
            self._lr_age += 1

    def _reduce_learning_rate(self, model):
        self._lr *= self._lr_reduction_factor
        model.optimizer.lr = self._lr
        print("\nnew lr:{}".format(self._lr))
        self._reset_learning_rate_age()

    def _fit(self, model, training_set, validation_set, initial_epoch):
        fit_fn_params = {
            'x': training_set,
            'initial_epoch': initial_epoch,
            'epochs': initial_epoch + 1,
            'validation_data': validation_set
        }
        if self._class_weights is not None:
            fit_fn_params['class_weight'] = self._class_weights
        history = model.fit(**fit_fn_params)
        new_val_loss = history.history['val_loss'][0]
        return new_val_loss

    def _is_learning_rate_to_reduce(self):
        return not self._is_early_stopping() and \
               not self._static_lr and \
               self._best_val_loss_age >= self._lr_reducer_patience and \
               self._lr_age >= self._lr_reducer_patience

    def _reset_learning_rate_age(self):
        self._lr_age = 0

    def _is_early_stopping(self):
        return self._best_val_loss_age >= self._early_stopping_patience
