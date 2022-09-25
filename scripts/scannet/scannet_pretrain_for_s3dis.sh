#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

# TRAIN
python main_instance_segmentation.py \
general.experiment_name="pretrain_for_s3dis" \
data.train_mode=train_validation