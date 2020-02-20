#!/bin/sh
wget -O data/raw_data.zip https://physionet.org/files/challenge-2017/1.0.0/training2017.zip
unzip -o -d data/raw_data data/raw_data.zip
rm data/raw_data.zip
wget -O data/raw_data/labels.csv https://physionet.org/files/challenge-2017/1.0.0/REFERENCE-v3.csv