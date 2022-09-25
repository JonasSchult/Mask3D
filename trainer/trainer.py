import gc
from contextlib import nullcontext
from pathlib import Path
import statistics
import shutil
import os
import math
import pyviz3d.visualizer as vis
from torch_scatter import scatter_mean
import matplotlib
from benchmark.evaluate_semantic_instance import evaluate
from collections import defaultdict
from sklearn.cluster import DBSCAN
from utils.votenet_utils.eval_det import eval_det
from datasets.scannet200.scannet200_splits import HEAD_CATS_SCANNET_200, TAIL_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, VALID_CLASS_IDS_200_VALIDATION

import hydra
import MinkowskiEngine as ME
import numpy as np
import pytorch_lightning as pl
import torch
from models.metrics import IoU
import random
import colorsys
from typing import List, Tuple
import functools


@functools.lru_cache(20)
def get_evenly_distributed_colors(count: int) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    # lru cache caches color tuples
    HSV_tuples = [(x/count, 1.0, 1.0) for x in range(count)]
    random.shuffle(HSV_tuples)
    return list(map(lambda x: (np.array(colorsys.hsv_to_rgb(*x))*255).astype(np.uint8), HSV_tuples))

class RegularCheckpointing(pl.Callback):
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        general = pl_module.config.general
        trainer.save_checkpoint(f"{general.save_dir}/last-epoch.ckpt")
        print("Checkpoint created")

class InstanceSegmentation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.decoder_id = config.general.decoder_id

        if config.model.train_on_segments:
            self.mask_type = "segment_mask"
        else:
            self.mask_type = "masks"

        self.eval_on_segments = config.general.eval_on_segments

        self.config = config
        self.save_hyperparameters()
        # model
        self.model = hydra.utils.instantiate(config.model)
        self.optional_freeze = nullcontext
        if config.general.freeze_backbone:
            self.optional_freeze = torch.no_grad
        # loss
        self.ignore_label = config.data.ignore_label

        matcher = hydra.utils.instantiate(config.matcher)
        weight_dict = {"loss_ce": matcher.cost_class,
                       "loss_mask": matcher.cost_mask,
                       "loss_dice": matcher.cost_dice}

        aux_weight_dict = {}
        for i in range(self.model.num_levels * self.model.num_decoders):
            if i not in self.config.general.ignore_mask_idx:
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            else:
                aux_weight_dict.update({k + f"_{i}": 0. for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

        self.preds = dict()
        self.bbox_preds = dict()
        self.bbox_gt = dict()

        self.criterion = hydra.utils.instantiate(config.loss, matcher=matcher, weight_dict=weight_dict)

        # metrics
        self.confusion = hydra.utils.instantiate(config.metrics)
        self.iou = IoU()
        # misc
        self.labels_info = dict()

    def forward(self, x, point2segment=None, raw_coordinates=None, is_eval=False):
        with self.optional_freeze():
            x = self.model(x, point2segment, raw_coordinates=raw_coordinates,
                           is_eval=is_eval)
        return x

    def training_step(self, batch, batch_idx):
        data, target, file_names = batch

        if data.features.shape[0] > self.config.general.max_batch_size:
            print("data exceeds threshold")
            raise RuntimeError("BATCH TOO BIG")

        if len(target) == 0:
            print("no targets")
            return None

        raw_coordinates = None
        if self.config.data.add_raw_coordinates:
            raw_coordinates = data.features[:, -3:]
            data.features = data.features[:, :-3]

        data = ME.SparseTensor(coordinates=data.coordinates,
                              features=data.features,
                              device=self.device)

        try:
            output = self.forward(data,
                                  point2segment=[target[i]['point2segment'] for i in range(len(target))],
                                  raw_coordinates=raw_coordinates)
        except RuntimeError as run_err:
            print(run_err)
            if 'only a single point gives nans in cross-attention' == run_err.args[0]:
                return None
            else:
                raise run_err

        try:
            losses = self.criterion(output, target, mask_type=self.mask_type)
        except ValueError as val_err:
            print(f"ValueError: {val_err}")
            print(f"data shape: {data.shape}")
            print(f"data feat shape:  {data.features.shape}")
            print(f"data feat nans:   {data.features.isnan().sum()}")
            print(f"output: {output}")
            print(f"target: {target}")
            print(f"filenames: {file_names}")
            raise val_err

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        logs = {f"train_{k}": v.detach().cpu().item() for k,v in losses.items()}

        logs['train_mean_loss_ce'] = statistics.mean([item for item in [v for k, v in logs.items() if "loss_ce" in k]])

        logs['train_mean_loss_mask'] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_mask" in k]])

        logs['train_mean_loss_dice'] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_dice" in k]])

        self.log_dict(logs)
        return sum(losses.values())

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def export(self, pred_masks, scores, pred_classes, file_names, decoder_id):
        root_path = f"eval_output"
        base_path = f"{root_path}/instance_evaluation_{self.config.general.experiment_name}_{self.current_epoch}/decoder_{decoder_id}"
        pred_mask_path = f"{base_path}/pred_mask"

        Path(pred_mask_path).mkdir(parents=True, exist_ok=True)

        file_name = file_names
        with open(f"{base_path}/{file_name}.txt", "w") as fout:
            real_id = -1
            for instance_id in range(len(pred_classes)):
                real_id += 1
                pred_class = pred_classes[instance_id]
                score = scores[instance_id]
                mask = pred_masks[:, instance_id].astype("uint8")

                if score > self.config.general.export_threshold:
                    # reduce the export size a bit. I guess no performance difference
                    np.savetxt(f"{pred_mask_path}/{file_name}_{real_id}.txt", mask, fmt="%d")
                    fout.write(f"pred_mask/{file_name}_{real_id}.txt {pred_class} {score}\n")

    def training_epoch_end(self, outputs):
        train_loss = sum([out["loss"].cpu().item() for out in outputs]) / len(outputs)
        results = {"train_loss_mean": train_loss}
        self.log_dict(results)

    def validation_epoch_end(self, outputs):
        self.test_epoch_end(outputs)

    def save_visualizations(self, target_full, full_res_coords,
                            sorted_masks, sort_classes, file_name, original_colors, original_normals,
                            sort_scores_values, point_size=20, sorted_heatmaps=None,
                            query_pos=None, backbone_features=None):

        full_res_coords -= full_res_coords.mean(axis=0)

        gt_pcd_pos = []
        gt_pcd_normals = []
        gt_pcd_color = []
        gt_inst_pcd_color = []
        gt_boxes = []

        if 'labels' in target_full:
            instances_colors = torch.from_numpy(
                np.vstack(get_evenly_distributed_colors(target_full['labels'].shape[0])))
            for instance_counter, (label, mask) in enumerate(zip(target_full['labels'], target_full['masks'])):
                if label == 255:
                    continue

                mask_tmp = mask.detach().cpu().numpy()
                mask_coords = full_res_coords[mask_tmp.astype(bool), :]

                if len(mask_coords) == 0:
                    continue

                gt_pcd_pos.append(mask_coords)
                mask_coords_min = full_res_coords[mask_tmp.astype(bool), :].min(axis=0)
                mask_coords_max = full_res_coords[mask_tmp.astype(bool), :].max(axis=0)
                size = mask_coords_max - mask_coords_min
                mask_coords_middle = mask_coords_min + size / 2

                gt_boxes.append({"position": mask_coords_middle, "size": size,
                                 "color": self.validation_dataset.map2color([label])[0]})

                gt_pcd_color.append(
                    self.validation_dataset.map2color([label]).repeat(gt_pcd_pos[-1].shape[0], 1)
                )
                gt_inst_pcd_color.append(instances_colors[instance_counter % len(instances_colors)].unsqueeze(0).repeat(gt_pcd_pos[-1].shape[0], 1))

                gt_pcd_normals.append(original_normals[mask_tmp.astype(bool), :])

            gt_pcd_pos = np.concatenate(gt_pcd_pos)
            gt_pcd_normals = np.concatenate(gt_pcd_normals)
            gt_pcd_color = np.concatenate(gt_pcd_color)
            gt_inst_pcd_color = np.concatenate(gt_inst_pcd_color)

        v = vis.Visualizer()

        v.add_points("RGB Input", full_res_coords,
                     colors=original_colors,
                     normals=original_normals,
                     visible=True,
                     point_size=point_size)

        if backbone_features is not None:
            v.add_points("PCA", full_res_coords,
                         colors=backbone_features,
                         normals=original_normals,
                         visible=False,
                         point_size=point_size)

        if 'labels' in target_full:
            v.add_points("Semantics (GT)", gt_pcd_pos,
                         colors=gt_pcd_color,
                         normals=gt_pcd_normals,
                         alpha=0.8,
                         visible=False,
                         point_size=point_size)
            v.add_points("Instances (GT)", gt_pcd_pos,
                         colors=gt_inst_pcd_color,
                         normals=gt_pcd_normals,
                         alpha=0.8,
                         visible=False,
                         point_size=point_size)

        pred_coords = []
        pred_normals = []
        pred_sem_color = []
        pred_inst_color = []

        for did in range(len(sorted_masks)):
            instances_colors = torch.from_numpy(
                np.vstack(get_evenly_distributed_colors(max(1, sorted_masks[did].shape[1]))))

            for i in reversed(range(sorted_masks[did].shape[1])):
                coords = full_res_coords[sorted_masks[did][:, i].astype(bool), :]

                mask_coords = full_res_coords[sorted_masks[did][:,i].astype(bool), :]
                mask_normals = original_normals[sorted_masks[did][:,i].astype(bool), :]

                label = sort_classes[did][i]

                if len(mask_coords) == 0:
                    continue

                pred_coords.append(mask_coords)
                pred_normals.append(mask_normals)

                pred_sem_color.append(
                    self.validation_dataset.map2color([label]).repeat(
                        mask_coords.shape[0], 1)
                )

                pred_inst_color.append(instances_colors[i % len(instances_colors)].unsqueeze(0).repeat(
                    mask_coords.shape[0], 1)
                )

            if len(pred_coords) > 0:
                pred_coords = np.concatenate(pred_coords)
                pred_normals = np.concatenate(pred_normals)
                pred_sem_color = np.concatenate(pred_sem_color)
                pred_inst_color = np.concatenate(pred_inst_color)

                v.add_points("Semantics (Mask3D)", pred_coords,
                             colors=pred_sem_color,
                             normals=pred_normals,
                             visible=False,
                             alpha=0.8,
                             point_size=point_size)
                v.add_points("Instances (Mask3D)", pred_coords,
                             colors=pred_inst_color,
                             normals=pred_normals,
                             visible=False,
                             alpha=0.8,
                             point_size=point_size)

        v.save(f"{self.config['general']['save_dir']}/visualizations/{file_name}")

    def eval_step(self, batch, batch_idx):
        data, target, file_names = batch
        inverse_maps = data.inverse_maps
        target_full = data.target_full
        original_colors = data.original_colors
        data_idx = data.idx
        original_normals = data.original_normals
        original_coordinates = data.original_coordinates

        #if len(target) == 0 or len(target_full) == 0:
        #    print("no targets")
        #    return None

        if len(data.coordinates) == 0:
            return 0.

        raw_coordinates = None
        if self.config.data.add_raw_coordinates:
            raw_coordinates = data.features[:, -3:]
            data.features = data.features[:, :-3]

        if raw_coordinates.shape[0] == 0:
            return 0.

        data = ME.SparseTensor(coordinates=data.coordinates, features=data.features, device=self.device)


        try:
            output = self.forward(data,
                                  point2segment=[target[i]['point2segment'] for i in range(len(target))],
                                  raw_coordinates=raw_coordinates,
                                  is_eval=True)
        except RuntimeError as run_err:
            print(run_err)
            if 'only a single point gives nans in cross-attention' == run_err.args[0]:
                return None
            else:
                raise run_err

        if self.config.data.test_mode != "test":
            if self.config.trainer.deterministic:
                torch.use_deterministic_algorithms(False)

            try:
                losses = self.criterion(output, target,
                                        mask_type=self.mask_type)
            except ValueError as val_err:
                print(f"ValueError: {val_err}")
                print(f"data shape: {data.shape}")
                print(f"data feat shape:  {data.features.shape}")
                print(f"data feat nans:   {data.features.isnan().sum()}")
                print(f"output: {output}")
                print(f"target: {target}")
                print(f"filenames: {file_names}")
                raise val_err

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            if self.config.trainer.deterministic:
                torch.use_deterministic_algorithms(True)

        if self.config.general.save_visualizations:
            backbone_features = output['backbone_features'].F.detach().cpu().numpy()
            from sklearn import decomposition
            pca = decomposition.PCA(n_components=3)
            pca.fit(backbone_features)
            pca_features = pca.transform(backbone_features)
            rescaled_pca = 255 * (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())

        self.eval_instance_step(output, target, target_full, inverse_maps, file_names, original_coordinates,
                                original_colors, original_normals, raw_coordinates, data_idx,
                                backbone_features=rescaled_pca if self.config.general.save_visualizations else None)

        if self.config.data.test_mode != "test":
            return {f"val_{k}": v.detach().cpu().item() for k, v in losses.items()}
        else:
            return 0.

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def get_full_res_mask(self, mask, inverse_map, point2segment_full, is_heatmap=False):
        mask = mask.detach().cpu()[inverse_map]  # full res

        if self.eval_on_segments and is_heatmap==False:
            mask = scatter_mean(mask, point2segment_full, dim=0)  # full res segments
            mask = (mask > 0.5).float()
            mask = mask.detach().cpu()[point2segment_full.cpu()]  # full res points

        return mask


    def get_mask_and_scores(self, mask_cls, mask_pred, num_queries=100, num_classes=18, device=None):
        if device is None:
            device = self.device
        labels = torch.arange(num_classes, device=device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

        if self.config.general.topk_per_image != -1 :
            scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(self.config.general.topk_per_image, sorted=True)
        else:
            scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(num_queries, sorted=True)

        labels_per_query = labels[topk_indices]
        topk_indices = topk_indices // num_classes
        mask_pred = mask_pred[:, topk_indices]

        result_pred_mask = (mask_pred > 0).float()
        heatmap = mask_pred.float().sigmoid()

        mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (result_pred_mask.sum(0) + 1e-6)
        score = scores_per_query * mask_scores_per_image
        classes = labels_per_query

        return score, result_pred_mask, classes, heatmap

    def eval_instance_step(self, output, target_low_res, target_full_res, inverse_maps, file_names,
                           full_res_coords, original_colors, original_normals, raw_coords, idx, first_full_res=False,
                           backbone_features=None,):
        label_offset = self.validation_dataset.label_offset
        prediction = output['aux_outputs']
        prediction.append({
            'pred_logits': output['pred_logits'],
            'pred_masks': output['pred_masks']
        })

        prediction[self.decoder_id]['pred_logits'] = torch.functional.F.softmax(
            prediction[self.decoder_id]['pred_logits'],
            dim=-1)[..., :-1]

        all_pred_classes = list()
        all_pred_masks = list()
        all_pred_scores = list()
        all_heatmaps = list()
        all_query_pos = list()

        offset_coords_idx = 0
        for bid in range(len(prediction[self.decoder_id]['pred_masks'])):
            if not first_full_res:
                if self.model.train_on_segments:
                    masks = prediction[self.decoder_id]['pred_masks'][bid].detach().cpu()[target_low_res[bid]['point2segment'].cpu()]
                else:
                    masks = prediction[self.decoder_id]['pred_masks'][bid].detach().cpu()

                if self.config.general.use_dbscan:
                    new_preds = {
                        'pred_masks': list(),
                        'pred_logits': list(),
                    }

                    curr_coords_idx = masks.shape[0]
                    curr_coords = raw_coords[offset_coords_idx:curr_coords_idx + offset_coords_idx]
                    offset_coords_idx += curr_coords_idx

                    for curr_query in range(masks.shape[1]):
                        curr_masks = masks[:, curr_query] > 0

                        if curr_coords[curr_masks].shape[0] > 0:
                            clusters = DBSCAN(eps=self.config.general.dbscan_eps,
                                              min_samples=self.config.general.dbscan_min_points,
                                              n_jobs=-1).fit(curr_coords[curr_masks]).labels_

                            new_mask = torch.zeros(curr_masks.shape, dtype=int)
                            new_mask[curr_masks] = torch.from_numpy(clusters) + 1

                            for cluster_id in np.unique(clusters):
                                original_pred_masks = masks[:, curr_query]
                                if cluster_id != -1:
                                    new_preds['pred_masks'].append(original_pred_masks * (new_mask == cluster_id + 1))
                                    new_preds['pred_logits'].append(
                                        prediction[self.decoder_id]['pred_logits'][bid, curr_query])

                    scores, masks, classes, heatmap = self.get_mask_and_scores(
                        torch.stack(new_preds['pred_logits']).cpu(),
                        torch.stack(new_preds['pred_masks']).T,
                        len(new_preds['pred_logits']),
                        self.model.num_classes - 1)
                else:
                    scores, masks, classes, heatmap = self.get_mask_and_scores(
                    prediction[self.decoder_id]['pred_logits'][bid].detach().cpu(),
                    masks,
                    prediction[self.decoder_id]['pred_logits'][bid].shape[0],
                    self.model.num_classes - 1)

                masks = self.get_full_res_mask(masks,
                                               inverse_maps[bid],
                                               target_full_res[bid]['point2segment'])

                heatmap = self.get_full_res_mask(heatmap,
                                                 inverse_maps[bid],
                                                 target_full_res[bid]['point2segment'],
                                                 is_heatmap=True)

                if backbone_features is not None:
                    backbone_features = self.get_full_res_mask(torch.from_numpy(backbone_features),
                                                               inverse_maps[bid],
                                                               target_full_res[bid]['point2segment'],
                                                               is_heatmap=True)
                    backbone_features = backbone_features.numpy()
            else:
                assert False,  "not tested"
                masks = self.get_full_res_mask(prediction[self.decoder_id]['pred_masks'][bid].cpu(),
                                               inverse_maps[bid],
                                               target_full_res[bid]['point2segment'])

                scores, masks, classes, heatmap = self.get_mask_and_scores(
                    prediction[self.decoder_id]['pred_logits'][bid].cpu(),
                    masks,
                    prediction[self.decoder_id]['pred_logits'][bid].shape[0],
                    self.model.num_classes - 1,
                    device="cpu")

            masks = masks.numpy()
            heatmap = heatmap.numpy()

            sort_scores = scores.sort(descending=True)
            sort_scores_index = sort_scores.indices.cpu().numpy()
            sort_scores_values = sort_scores.values.cpu().numpy()
            sort_classes = classes[sort_scores_index]

            sorted_masks = masks[:, sort_scores_index]
            sorted_heatmap = heatmap[:, sort_scores_index]

            if self.config.general.filter_out_instances:
                keep_instances = set()
                pairwise_overlap = (sorted_masks.T @ sorted_masks)
                normalization = pairwise_overlap.max(axis=0)
                norm_overlaps = pairwise_overlap / normalization

                for instance_id in range(norm_overlaps.shape[0]):
                    # filter out unlikely masks and nearly empty masks
                    # if not(sort_scores_values[instance_id] < 0.3 or sorted_masks[:, instance_id].sum() < 500):
                    if not(sort_scores_values[instance_id] < self.config.general.scores_threshold):
                        # check if mask != empty
                        if not sorted_masks[:, instance_id].sum() == 0.0:
                            overlap_ids = set(np.nonzero(norm_overlaps[instance_id, :] > self.config.general.iou_threshold)[0])

                            if len(overlap_ids) == 0:
                                keep_instances.add(instance_id)
                            else:
                                if instance_id == min(overlap_ids):
                                    keep_instances.add(instance_id)

                keep_instances = sorted(list(keep_instances))
                all_pred_classes.append(sort_classes[keep_instances])
                all_pred_masks.append(sorted_masks[:, keep_instances])
                all_pred_scores.append(sort_scores_values[keep_instances])
                all_heatmaps.append(sorted_heatmap[:, keep_instances])
            else:
                all_pred_classes.append(sort_classes)
                all_pred_masks.append(sorted_masks)
                all_pred_scores.append(sort_scores_values)
                all_heatmaps.append(sorted_heatmap)

        if self.validation_dataset.dataset_name == "scannet200":
            all_pred_classes[bid][all_pred_classes[bid] == 0] = -1
            if self.config.data.test_mode != "test":
                target_full_res[bid]['labels'][target_full_res[bid]['labels'] == 0] = -1

        for bid in range(len(prediction[self.decoder_id]['pred_masks'])):
            all_pred_classes[bid] = self.validation_dataset._remap_model_output(all_pred_classes[bid].cpu() + label_offset)

            if self.config.data.test_mode != "test" and len(target_full_res) != 0:
                target_full_res[bid]['labels'] = self.validation_dataset._remap_model_output(
                    target_full_res[bid]['labels'].cpu() + label_offset)

                # PREDICTION BOX
                bbox_data = []
                for query_id in range(all_pred_masks[bid].shape[1]):  # self.model.num_queries
                    obj_coords = full_res_coords[bid][all_pred_masks[bid][:, query_id].astype(bool), :]
                    if obj_coords.shape[0] > 0:
                        obj_center = obj_coords.mean(axis=0)
                        obj_axis_length = obj_coords.max(axis=0) - obj_coords.min(axis=0)

                        bbox = np.concatenate((obj_center, obj_axis_length))

                        bbox_data.append((all_pred_classes[bid][query_id].item(), bbox,
                                          all_pred_scores[bid][query_id]
                        ))
                self.bbox_preds[file_names[bid]] = bbox_data

                # GT BOX
                bbox_data = []
                for obj_id in range(target_full_res[bid]['masks'].shape[0]):
                    if target_full_res[bid]['labels'][obj_id].item() == 255:
                        continue

                    obj_coords = full_res_coords[bid][target_full_res[bid]['masks'][obj_id, :].cpu().detach().numpy().astype(bool), :]
                    if obj_coords.shape[0] > 0:
                        obj_center = obj_coords.mean(axis=0)
                        obj_axis_length = obj_coords.max(axis=0) - obj_coords.min(axis=0)

                        bbox = np.concatenate((obj_center, obj_axis_length))
                        bbox_data.append((target_full_res[bid]['labels'][obj_id].item(), bbox))

                self.bbox_gt[file_names[bid]] = bbox_data

            if self.config.general.eval_inner_core == -1:
                self.preds[file_names[bid]] = {
                    'pred_masks': all_pred_masks[bid],
                    'pred_scores': all_pred_scores[bid],
                    'pred_classes': all_pred_classes[bid]
                }
            else:
                # prev val_dataset
                self.preds[file_names[bid]] = {
                    'pred_masks': all_pred_masks[bid][self.test_dataset.data[idx[bid]]['cond_inner']],
                    'pred_scores': all_pred_scores[bid],
                    'pred_classes': all_pred_classes[bid]
                }

            if self.config.general.save_visualizations:
                if 'cond_inner' in self.test_dataset.data[idx[bid]]:
                    target_full_res[bid]['masks'] = target_full_res[bid]['masks'][:, self.test_dataset.data[idx[bid]]['cond_inner']]
                    self.save_visualizations(target_full_res[bid],
                                             full_res_coords[bid][self.test_dataset.data[idx[bid]]['cond_inner']],
                                             [self.preds[file_names[bid]]['pred_masks']],
                                             [self.preds[file_names[bid]]['pred_classes']],
                                             file_names[bid],
                                             original_colors[bid][self.test_dataset.data[idx[bid]]['cond_inner']],
                                             original_normals[bid][self.test_dataset.data[idx[bid]]['cond_inner']],
                                             [self.preds[file_names[bid]]['pred_scores']],
                                             sorted_heatmaps=[all_heatmaps[bid][self.test_dataset.data[idx[bid]]['cond_inner']]],
                                             query_pos=all_query_pos[bid][self.test_dataset.data[idx[bid]]['cond_inner']] if len(all_query_pos) > 0 else None,
                                             backbone_features=backbone_features[self.test_dataset.data[idx[bid]]['cond_inner']],
                                             point_size=self.config.general.visualization_point_size)
                else:
                    self.save_visualizations(target_full_res[bid],
                                             full_res_coords[bid],
                                             [self.preds[file_names[bid]]['pred_masks']],
                                             [self.preds[file_names[bid]]['pred_classes']],
                                             file_names[bid],
                                             original_colors[bid],
                                             original_normals[bid],
                                             [self.preds[file_names[bid]]['pred_scores']],
                                             sorted_heatmaps=[all_heatmaps[bid]],
                                             query_pos=all_query_pos[bid] if len(all_query_pos) > 0 else None,
                                             backbone_features=backbone_features,
                                             point_size=self.config.general.visualization_point_size)

            if self.config.general.export:
                if self.validation_dataset.dataset_name == "stpls3d":
                    scan_id, _, _, crop_id = file_names[bid].split("_")
                    crop_id = int(crop_id.replace(".txt", ""))
                    file_name = f"{scan_id}_points_GTv3_0{crop_id}_inst_nostuff"

                    self.export(
                        self.preds[file_names[bid]]['pred_masks'],
                        self.preds[file_names[bid]]['pred_scores'],
                        self.preds[file_names[bid]]['pred_classes'],
                        file_name,
                        self.decoder_id
                    )
                else:
                    self.export(
                        self.preds[file_names[bid]]['pred_masks'],
                        self.preds[file_names[bid]]['pred_scores'],
                        self.preds[file_names[bid]]['pred_classes'],
                        file_names[bid],
                        self.decoder_id
                    )

    def eval_instance_epoch_end(self):
        log_prefix = f"val"
        ap_results = {}

        head_results, tail_results, common_results = [], [], []


        box_ap_50 = eval_det(self.bbox_preds, self.bbox_gt, ovthresh=0.5, use_07_metric=False)
        box_ap_25 = eval_det(self.bbox_preds, self.bbox_gt, ovthresh=0.25, use_07_metric=False)
        mean_box_ap_25 = sum([v for k, v in box_ap_25[-1].items()]) / len(box_ap_25[-1].keys())
        mean_box_ap_50 = sum([v for k, v in box_ap_50[-1].items()]) / len(box_ap_50[-1].keys())

        ap_results[f"{log_prefix}_mean_box_ap_25"] = mean_box_ap_25
        ap_results[f"{log_prefix}_mean_box_ap_50"] = mean_box_ap_50

        for class_id in box_ap_50[-1].keys():
            class_name = self.train_dataset.label_info[class_id]['name']
            ap_results[f"{log_prefix}_{class_name}_val_box_ap_50"] = box_ap_50[-1][class_id]

        for class_id in box_ap_25[-1].keys():
            class_name = self.train_dataset.label_info[class_id]['name']
            ap_results[f"{log_prefix}_{class_name}_val_box_ap_25"] = box_ap_25[-1][class_id]

        root_path = f"eval_output"
        base_path = f"{root_path}/instance_evaluation_{self.config.general.experiment_name}_{self.current_epoch}"

        if self.validation_dataset.dataset_name in ["scannet", "stpls3d", "scannet200"]:
            gt_data_path = f"{self.validation_dataset.data_dir[0]}/instance_gt/{self.validation_dataset.mode}"
        else:
            gt_data_path = f"{self.validation_dataset.data_dir[0]}/instance_gt/Area_{self.config.general.area}"

        pred_path = f"{base_path}/tmp_output.txt"

        log_prefix = f"val"

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        try:
            if self.validation_dataset.dataset_name == "s3dis":
                new_preds = {}
                for key in self.preds.keys():
                    new_preds[key.replace(f"Area_{self.config.general.area}_", "")] = {
                        'pred_classes': self.preds[key]['pred_classes'] + 1,
                        'pred_masks': self.preds[key]['pred_masks'],
                        'pred_scores': self.preds[key]['pred_scores']
                    }
                mprec, mrec = evaluate(new_preds, gt_data_path, pred_path, dataset="s3dis")
                ap_results[f"{log_prefix}_mean_precision"] = mprec
                ap_results[f"{log_prefix}_mean_recall"] = mrec
            elif self.validation_dataset.dataset_name == "stpls3d":
                new_preds = {}
                for key in self.preds.keys():
                    new_preds[key.replace(".txt", "")] = {
                        'pred_classes': self.preds[key]['pred_classes'],
                        'pred_masks': self.preds[key]['pred_masks'],
                        'pred_scores': self.preds[key]['pred_scores']
                    }

                evaluate(new_preds, gt_data_path, pred_path, dataset="stpls3d")
            else:
                evaluate(self.preds, gt_data_path, pred_path, dataset=self.validation_dataset.dataset_name)
            with open(pred_path, "r") as fin:
                for line_id, line in enumerate(fin):
                    if line_id == 0:
                        # ignore header
                        continue
                    class_name, _, ap, ap_50, ap_25 = line.strip().split(",")

                    if self.validation_dataset.dataset_name == "scannet200":
                        if class_name in VALID_CLASS_IDS_200_VALIDATION:
                            ap_results[f"{log_prefix}_{class_name}_val_ap"] = float(ap)
                            ap_results[f"{log_prefix}_{class_name}_val_ap_50"] = float(ap_50)
                            ap_results[f"{log_prefix}_{class_name}_val_ap_25"] = float(ap_25)

                            if class_name in HEAD_CATS_SCANNET_200:
                                head_results.append(np.array((float(ap), float(ap_50), float(ap_25))))
                            elif class_name in COMMON_CATS_SCANNET_200:
                                common_results.append(np.array((float(ap), float(ap_50), float(ap_25))))
                            elif class_name in TAIL_CATS_SCANNET_200:
                                tail_results.append(np.array((float(ap), float(ap_50), float(ap_25))))
                            else:
                                assert(False, 'class not known!')
                    else:
                        ap_results[f"{log_prefix}_{class_name}_val_ap"] = float(ap)
                        ap_results[f"{log_prefix}_{class_name}_val_ap_50"] = float(ap_50)
                        ap_results[f"{log_prefix}_{class_name}_val_ap_25"] = float(ap_25)

            if self.validation_dataset.dataset_name == "scannet200":
                head_results = np.stack(head_results)
                common_results = np.stack(common_results)
                tail_results = np.stack(tail_results)

                # COULD BE NAN IF VOXEL SIZE TOO LARGE (MOUSE, FIRE ALARM, LIGHT SWITCH, POWER STRIP)
                head_results[np.isnan(head_results)] = 0.
                common_results[np.isnan(common_results)] = 0.
                tail_results[np.isnan(tail_results)] = 0.

                mean_tail_results = np.mean(tail_results, axis=0)
                mean_common_results = np.mean(common_results, axis=0)
                mean_head_results = np.mean(head_results, axis=0)

                ap_results[f"{log_prefix}_mean_tail_ap_25"] = mean_tail_results[0]
                ap_results[f"{log_prefix}_mean_common_ap_25"] = mean_common_results[0]
                ap_results[f"{log_prefix}_mean_head_ap_25"] = mean_head_results[0]

                ap_results[f"{log_prefix}_mean_tail_ap_50"] = mean_tail_results[1]
                ap_results[f"{log_prefix}_mean_common_ap_50"] = mean_common_results[1]
                ap_results[f"{log_prefix}_mean_head_ap_50"] = mean_head_results[1]

                ap_results[f"{log_prefix}_mean_tail_ap_25"] = mean_tail_results[2]
                ap_results[f"{log_prefix}_mean_common_ap_25"] = mean_common_results[2]
                ap_results[f"{log_prefix}_mean_head_ap_25"] = mean_head_results[2]

                overall_ap_results = np.vstack((head_results, common_results, tail_results)).mean(axis=0)

                ap_results[f"{log_prefix}_mean_ap"] = overall_ap_results[0]
                ap_results[f"{log_prefix}_mean_ap_50"] = overall_ap_results[1]
                ap_results[f"{log_prefix}_mean_ap_25"] = overall_ap_results[2]

                ap_results = {key: 0. if math.isnan(score) else score for key, score in ap_results.items()}
            else:
                mean_ap = statistics.mean([item for key, item in ap_results.items() if key.endswith("val_ap")])
                mean_ap_50 = statistics.mean([item for key, item in ap_results.items() if key.endswith("val_ap_50")])
                mean_ap_25 = statistics.mean([item for key, item in ap_results.items() if key.endswith("val_ap_25")])

                ap_results[f"{log_prefix}_mean_ap"] = mean_ap
                ap_results[f"{log_prefix}_mean_ap_50"] = mean_ap_50
                ap_results[f"{log_prefix}_mean_ap_25"] = mean_ap_25

                ap_results = {key: 0. if math.isnan(score) else score for key, score in ap_results.items()}
        except (IndexError, OSError) as e:
            print("NO SCORES!!!")
            ap_results[f"{log_prefix}_mean_ap"] = 0.
            ap_results[f"{log_prefix}_mean_ap_50"] = 0.
            ap_results[f"{log_prefix}_mean_ap_25"] = 0.

        self.log_dict(ap_results)

        if not self.config.general.export:
            shutil.rmtree(base_path)

        del self.preds
        del self.bbox_preds
        del self.bbox_gt

        gc.collect()

        self.preds = dict()
        self.bbox_preds = dict()
        self.bbox_gt = dict()

    def test_epoch_end(self, outputs):
        if self.config.general.export:
            return

        self.eval_instance_epoch_end()

        dd = defaultdict(list)
        for output in outputs:
            for key, val in output.items():  # .items() in Python 3.
                dd[key].append(val)

        dd = {k: statistics.mean(v) for k, v in dd.items()}

        dd['val_mean_loss_ce'] = statistics.mean([item for item in [v for k,v in dd.items() if "loss_ce" in k]])
        dd['val_mean_loss_mask'] = statistics.mean([item for item in [v for k,v in dd.items() if "loss_mask" in k]])
        dd['val_mean_loss_dice'] = statistics.mean([item for item in [v for k,v in dd.items() if "loss_dice" in k]])

        self.log_dict(dd)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.config.optimizer, params=self.parameters()
        )
        if "steps_per_epoch" in self.config.scheduler.scheduler.keys():
            self.config.scheduler.scheduler.steps_per_epoch = len(
                self.train_dataloader()
            )
        lr_scheduler = hydra.utils.instantiate(
            self.config.scheduler.scheduler, optimizer=optimizer
        )
        scheduler_config = {"scheduler": lr_scheduler}
        scheduler_config.update(self.config.scheduler.pytorch_lightning_params)
        return [optimizer], [scheduler_config]

    def prepare_data(self):
        self.train_dataset = hydra.utils.instantiate(self.config.data.train_dataset)
        self.validation_dataset = hydra.utils.instantiate(
            self.config.data.validation_dataset
        )
        self.test_dataset = hydra.utils.instantiate(self.config.data.test_dataset)
        self.labels_info = self.train_dataset.label_info

    def train_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.train_collation)
        return hydra.utils.instantiate(
            self.config.data.train_dataloader,
            self.train_dataset,
            collate_fn=c_fn,
        )

    def val_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        return hydra.utils.instantiate(
            self.config.data.validation_dataloader,
            self.validation_dataset,
            collate_fn=c_fn,
        )

    def test_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.test_collation)
        return hydra.utils.instantiate(
            self.config.data.test_dataloader,
            self.test_dataset,
            collate_fn=c_fn,
        )
