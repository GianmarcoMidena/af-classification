# Atrial Fibrillation classification with Convolutional Neural Networks (CNNs)

## Use
1) Download data
2) Partition data
3) Train models
4) Evaluate models

### Download data
```
./download_data.sh "data/raw_data"
```

### Partition data
```
python partition_data.py \
    -raw_data_folder "data/raw_data" \
    -labels_file_path "data/raw_data/labels.csv" \
    -sampling_frequency 300 \
    -test_set_size 528 \
    -n_splits 10 \
    -baseline_wander_removal_filter True \
    -output_dir "data/datasets/10_partitions"
```

### Train models
```
python train_model.py \
    -model_name "model1" \
    -data_dir "data/datasets/10_partitions" \
    -test_set_path "data/datasets/10_partitions/test.tf" \
    -n_records 5528 \
    -n_test_set_records 528 \
    -record_size 9000 \
    -n_folds 5 \
    -max_n_epochs 3 \
    -initial_lr 0.001 \
    -lr_reduction_factor 0.1 \
    -early_stopping_patience 15 \
    -lr_reducer_patience 5 \
    -batch_size 128 \
    -labels_file_path "data/raw_data/labels.csv" \
    -class_weighting True
    -weights_file_path "data/weights" \
    -output_model_path "models" \
    -fixed_fold 2 \
    -static_lr False
```

### Evaluate models
```
python evaluate_models.py \
    -models_paths "data/models/model1_fold_*.h5" "data/models/model2_fold_*.h5" "data/models/model3_fold_*.h5" \
    -model_names "model1" "model2" "model3" \
    -test_set_path "data/datasets/10_partitions/test.tf" \
    -record_size 9000 \
    -batch_size 528 \
    -output_plot_dir "plots"
```

## Dependencies
- Python 3.6.9
- Tensorflow 2.1.0
- Pandas 1.0.1
- Numpy 1.18.1