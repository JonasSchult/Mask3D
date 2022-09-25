#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_DBSCAN=0.95
CURR_TOPK=500
CURR_QUERY=150

# TRAIN
python main_instance_segmentation.py \
general.experiment_name="validation" \
general.eval_on_segments=true \
general.train_on_segments=true

# TEST
python main_instance_segmentation.py \
general.experiment_name="validation_query_${CURR_QUERY}_topk_${CURR_TOPK}_dbscan_${CURR_DBSCAN}" \
general.project_name="scannet_eval" \
general.checkpoint='checkpoints/scannet/scannet_val.ckpt' \
general.train_mode=false \
general.eval_on_segments=true \
general.train_on_segments=true \
model.num_queries=${CURR_QUERY} \
general.topk_per_image=${CURR_TOPK} \
general.use_dbscan=true \
general.dbscan_eps=${CURR_DBSCAN}
