class KFoldSplitter:
    def __init__(self, n_splits):
        self._n_splits = n_splits

    @property
    def n_splits(self):
        return self._n_splits

    def split(self, dataset_paths, fixed_fold=None):
        if (len(dataset_paths) % self._n_splits) != 0:
            raise Exception(f'the number of data files ({len(dataset_paths)}) is not divisible by k={k}')

        dataset_paths.sort()
        n_data_files_per_unit = int(len(dataset_paths) // self._n_splits)
        unit_start_indices = range(0, len(dataset_paths), n_data_files_per_unit)
        if (fixed_fold is not None) and \
                (fixed_fold >= 1) and (fixed_fold <= len(unit_start_indices)):
            unit_start_indices = [unit_start_indices[fixed_fold - 1]]
        for i in unit_start_indices:
            validation_set_path = dataset_paths[i:i + n_data_files_per_unit + 1]
            training_set_paths = dataset_paths[0:i] + dataset_paths[i + n_data_files_per_unit + 1:]
            yield training_set_paths, validation_set_path
