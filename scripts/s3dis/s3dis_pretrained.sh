#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_AREA=1  # set the area number accordingly [1,6]
CURR_DBSCAN=0.6
CURR_TOPK=-1
CURR_QUERY=100

python main_instance_segmentation.py \
  general.project_name="s3dis" \
  general.experiment_name="area${CURR_AREA}_pretrained" \
  data.batch_size=4 \
  data/datasets=s3dis \
  general.num_targets=14 \
  data.num_labels=13 \
  general.area=${CURR_AREA} \
  general.checkpoint="checkpoints/s3dis/scannet_pretrained/scannet_pretrained.ckpt" \
  trainer.check_val_every_n_epoch=10 \
  optimizer.lr=0.00001

python main_instance_segmentation.py \
general.project_name="s3dis_eval" \
general.experiment_name="area${CURR_AREA}_pretrained_eps_${CURR_DBSCAN}_topk_${CURR_TOPK}_q_${CURR_QUERY}" \
general.checkpoint="checkpoints/s3dis/scannet_pretrained/area${CURR_AREA}.ckpt" \
general.train_mode=false \
data.batch_size=4 \
data/datasets=s3dis \
general.num_targets=14 \
data.num_labels=13 \
general.area=${CURR_AREA} \
model.num_queries=${CURR_QUERY} \
general.topk_per_image=${CURR_TOPK} \
general.use_dbscan=true \
general.dbscan_eps=${CURR_DBSCAN}
