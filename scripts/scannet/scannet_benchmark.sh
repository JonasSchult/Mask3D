#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_DBSCAN=0.95
CURR_TOPK=300
CURR_QUERY=150

# TRAIN
python main_instance_segmentation.py \
general.experiment_name="benchmark" \
general.eval_on_segments=true \
general.train_on_segments=true \
data.train_mode=train_validation

# TEST
python main_instance_segmentation.py \
general.experiment_name="benchmark_query_${CURR_QUERY}_topk_${CURR_TOPK}_dbscan_${CURR_DBSCAN}" \
general.project_name="scannet_eval" \
general.checkpoint='checkpoints/scannet/scannet_benchmark.ckpt' \
general.eval_on_segments=true \
general.train_on_segments=true \
general.train_mode=false \
general.export=true \
data.test_mode=test \
model.num_queries=${CURR_QUERY} \
general.topk_per_image=${CURR_TOPK} \
general.use_dbscan=true \
general.dbscan_eps=${CURR_DBSCAN}
