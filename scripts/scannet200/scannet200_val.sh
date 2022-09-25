#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_DBSCAN=0.95
CURR_TOPK=750
CURR_QUERY=150

# TRAIN
python main_instance_segmentation.py \
general.experiment_name="scannet200_val" \
general.project_name="scannet200" \
data/datasets=scannet200 \
general.num_targets=201 \
data.num_labels=200 \
general.scannet200=true \
general.eval_on_segments=true \
general.train_on_segments=true

# TEST
python main_instance_segmentation.py \
general.experiment_name="scannet200_val_query_${CURR_QUERY}_topk_${CURR_TOPK}_dbscan_${CURR_DBSCAN}" \
general.project_name="scannet200_eval" \
general.checkpoint="checkpoints/scannet200/scannet200_val.ckpt" \
data/datasets=scannet200 \
general.num_targets=201 \
data.num_labels=200 \
general.scannet200=true \
general.eval_on_segments=true \
general.train_on_segments=true \
general.train_mode=false \
model.num_queries=${CURR_QUERY} \
general.topk_per_image=${CURR_TOPK} \
general.use_dbscan=true \
general.dbscan_eps=${CURR_DBSCAN}
