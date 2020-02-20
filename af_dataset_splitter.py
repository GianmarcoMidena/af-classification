from dataset_splitter import DatasetSplitter
from k_fold_splitter import KFoldSplitter
from af_dataset_builder import AFDatasetBuilder


class AFDatasetSplitter:
    def __init__(self, n_splits, full_train_set_paths, test_set_paths, batch_size, n_records, n_test_set_records,
                 target_record_len, seed=None):
        data_splitter = KFoldSplitter(n_splits)
        dataset_builder = AFDatasetBuilder(target_record_len=target_record_len)
        self._dataset_splitter = DatasetSplitter(splitter=data_splitter, dataset_builder=dataset_builder,
                                                 full_train_set_paths=full_train_set_paths,
                                                 test_set_paths=test_set_paths, batch_size=batch_size,
                                                 n_records=n_records, n_test_set_records=n_test_set_records, seed=seed)

    def split(self, fixed_fold=None):
        return self._dataset_splitter.split(fixed_fold=fixed_fold)
