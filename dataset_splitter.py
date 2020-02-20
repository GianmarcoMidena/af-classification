class DatasetSplitter:
    def __init__(self, splitter, dataset_builder, full_train_set_paths, test_set_paths, batch_size, n_records,
                 n_test_set_records, seed=None):
        self._splitter = splitter
        self._dataset_builder = dataset_builder
        self._full_train_set_paths = full_train_set_paths
        self._test_set_paths = test_set_paths
        self._batch_size = batch_size
        self._n_records = n_records
        self._n_test_set_records = n_test_set_records
        self._seed = seed

    def split(self, fixed_fold=None):
        if fixed_fold is None:
            split = self._splitter.split(self._full_train_set_paths)
        else:
            split = self._splitter.split(self._full_train_set_paths, fixed_fold)

        n_train_set_records = self._count_train_set_records(self._n_records, self._n_test_set_records)

        for train_set_paths, validation_set_paths in split:
            training_set = self._dataset_builder.get_training_set(paths=train_set_paths, n_records=n_train_set_records,
                                                                  batch_size=self._batch_size, seed=self._seed)
            validation_set = self._dataset_builder.get_validation_set(paths=validation_set_paths,
                                                                      batch_size=self._batch_size, seed=self._seed)
            test_set = self._dataset_builder.get_test_set(paths=self._test_set_paths,
                                                          batch_size=self._n_test_set_records, seed=self._seed)
            yield training_set, validation_set, test_set

    def _count_train_set_records(self, n_records, n_test_set_records):
        n_full_train_set_records = n_records - n_test_set_records
        n_splits = self._splitter.n_splits
        n_records_per_file = int(n_full_train_set_records // n_splits)
        n_train_set_records = n_records_per_file * (n_splits - 1)
        return n_train_set_records
