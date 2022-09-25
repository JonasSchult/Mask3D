#!/bin/bash
export OMP_NUM_THREADS=3

CURR_DBSCAN=12.5
CURR_TOPK=200
CURR_QUERY=160
CURR_SIZE=54
CURR_THRESHOLD=0.01

# TRAIN network 1 with voxel size 0.333
python main_instance_segmentation.py \
general.experiment_name="benchmark_03" \
general.project_name="stpls3d" \
data/datasets=stpls3d \
general.num_targets=15 \
data.num_labels=15 \
data.voxel_size=0.333 \
data.num_workers=10 \
data.cache_data=true \
data.cropping_v1=false \
general.reps_per_epoch=100 \
model.num_queries=${CURR_QUERY} \
general.on_crops=true \
model.config.backbone._target_=models.Res16UNet18B \
data.crop_length=${CURR_SIZE} \
general.eval_inner_core=50.0 \
data.train_mode=train_validation

# TRAIN network 2 with voxel size 0.2 and larger backbone
python main_instance_segmentation.py \
general.experiment_name="benchmark_02" \
general.project_name="stpls3d" \
data/datasets=stpls3d \
general.num_targets=15 \
data.num_labels=15 \
data.voxel_size=0.2 \
data.num_workers=10 \
data.cache_data=true \
data.cropping_v1=false \
general.reps_per_epoch=100 \
model.num_queries=${CURR_QUERY} \
general.on_crops=true \
data.crop_length=${CURR_SIZE} \
general.eval_inner_core=50.0 \
data.train_mode=train_validation

# TEST network 1
python main_instance_segmentation.py \
general.experiment_name="benchmark_03_query_${CURR_QUERY}_topk_${CURR_TOPK}_dbscan_${CURR_DBSCAN}_size_${CURR_SIZE}_T_${CURR_THRESHOLD}" \
general.project_name="stpls3d_eval" \
data/datasets=stpls3d \
general.num_targets=15 \
data.num_labels=15 \
data.voxel_size=0.333 \
data.num_workers=10 \
data.cache_data=true \
data.cropping_v1=false \
general.reps_per_epoch=100 \
model.num_queries=${CURR_QUERY} \
general.on_crops=true \
model.config.backbone._target_=models.Res16UNet18B \
general.train_mode=false \
general.checkpoint="checkpoints/stpls3d/stpls3d_benchmark_03.ckpt" \
data.crop_length=${CURR_SIZE} \
general.eval_inner_core=50.0 \
general.topk_per_image=${CURR_TOPK} \
general.use_dbscan=true \
general.dbscan_eps=${CURR_DBSCAN} \
data.test_mode=test \
general.export=true

# TEST network 2
python main_instance_segmentation.py \
general.experiment_name="benchmark_02_query_${CURR_QUERY}_topk_${CURR_TOPK}_dbscan_${CURR_DBSCAN}_size_${CURR_SIZE}_T_${CURR_THRESHOLD}" \
general.project_name="stpls3d_eval" \
data/datasets=stpls3d \
general.num_targets=15 \
data.num_labels=15 \
data.voxel_size=0.2 \
data.num_workers=10 \
data.cache_data=true \
data.cropping_v1=false \
general.reps_per_epoch=100 \
model.num_queries=${CURR_QUERY} \
general.on_crops=true \
general.train_mode=false \
general.checkpoint="checkpoints/stpls3d/stpls3d_benchmark_02.ckpt" \
data.crop_length=${CURR_SIZE} \
general.eval_inner_core=50.0 \
general.topk_per_image=${CURR_TOPK} \
general.use_dbscan=true \
general.dbscan_eps=${CURR_DBSCAN} \
data.test_mode=test \
general.export=true

# COMBINE OUTPUTS OF ENSEMBLE
# VOXEL SIZE 0.2 FOR OBJECTS OF SMALL CLASSES; VOXEL SIZE 0.333 FOR OBJECTS OF LARGE CLASS CATEGORIES
# TODO FILL IN PATHS
python merge_exports.py
