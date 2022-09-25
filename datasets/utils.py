import MinkowskiEngine as ME
import numpy as np
import torch
from random import random

class VoxelizeCollate:
    def __init__(
            self,
            ignore_label=255,
            voxel_size=1,
            mode="test",
            small_crops=False,
            very_small_crops=False,
            batch_instance=False,
            probing=False,
            task="instance_segmentation",
            ignore_class_threshold=100,
            filter_out_classes=[],
            label_offset=0,
            num_queries=None
    ):
        assert task in ["instance_segmentation", "semantic_segmentation"], "task not known"
        self.task = task
        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset
        self.voxel_size = voxel_size
        self.ignore_label = ignore_label
        self.mode = mode
        self.batch_instance = batch_instance
        self.small_crops = small_crops
        self.very_small_crops = very_small_crops
        self.probing = probing
        self.ignore_class_threshold = ignore_class_threshold

        self.num_queries = num_queries

    def __call__(self, batch):
        if ("train" in self.mode) and (self.small_crops or self.very_small_crops):
            batch = make_crops(batch)
        if ("train" in self.mode) and self.very_small_crops:
            batch = make_crops(batch)
        return voxelize(batch, self.ignore_label, self.voxel_size, self.probing, self.mode,
                        task=self.task, ignore_class_threshold=self.ignore_class_threshold,
                        filter_out_classes=self.filter_out_classes, label_offset=self.label_offset,
                        num_queries=self.num_queries)


class VoxelizeCollateMerge:
    def __init__(
            self,
            ignore_label=255,
            voxel_size=1,
            mode="test",
            scenes=2,
            small_crops=False,
            very_small_crops=False,
            batch_instance=False,
            make_one_pc_noise=False,
            place_nearby=False,
            place_far=False,
            proba=1,
            probing=False,
            task="instance_segmentation"
    ):
        assert task in ["instance_segmentation", "semantic_segmentation"], "task not known"
        self.task = task
        self.mode = mode
        self.scenes = scenes
        self.small_crops = small_crops
        self.very_small_crops = very_small_crops
        self.ignore_label = ignore_label
        self.voxel_size = voxel_size
        self.batch_instance = batch_instance
        self.make_one_pc_noise = make_one_pc_noise
        self.place_nearby = place_nearby
        self.place_far = place_far
        self.proba = proba
        self.probing = probing

    def __call__(self, batch):
        if (
                ("train" in self.mode)
                and (not self.make_one_pc_noise)
                and (self.proba > random())
        ):
            if self.small_crops or self.very_small_crops:
                batch = make_crops(batch)
            if self.very_small_crops:
                batch = make_crops(batch)
            if self.batch_instance:
                batch = batch_instances(batch)
            new_batch = []
            for i in range(0, len(batch), self.scenes):
                batch_coordinates = []
                batch_features = []
                batch_labels = []

                batch_filenames = ""
                batch_raw_color = []
                batch_raw_normals = []

                offset_instance_id = 0
                offset_segment_id = 0

                for j in range(min(len(batch[i:]), self.scenes)):
                    batch_coordinates.append(batch[i + j][0])
                    batch_features.append(batch[i + j][1])

                    if j == 0:
                        batch_filenames = batch[i + j][3]
                    else:
                        batch_filenames = batch_filenames + f"+{batch[i + j][3]}"

                    batch_raw_color.append(batch[i + j][4])
                    batch_raw_normals.append(batch[i + j][5])

                    # make instance ids and segment ids unique
                    # take care that -1 instances stay at -1
                    batch_labels.append(batch[i + j][2] + [0, offset_instance_id, offset_segment_id])
                    batch_labels[-1][batch[i + j][2][:, 1] == -1, 1] = -1

                    max_instance_id, max_segment_id = batch[i + j][2].max(axis=0)[1:]
                    offset_segment_id = offset_segment_id + max_segment_id + 1
                    offset_instance_id = offset_instance_id + max_instance_id + 1

                if (len(batch_coordinates) == 2) and self.place_nearby:
                    border = batch_coordinates[0][:, 0].max()
                    border -= batch_coordinates[1][:, 0].min()
                    batch_coordinates[1][:, 0] += border
                elif (len(batch_coordinates) == 2) and self.place_far:
                    batch_coordinates[1] += (
                            np.random.uniform((-10, -10, -10), (10, 10, 10)) * 200
                    )
                new_batch.append(
                    (
                        np.vstack(batch_coordinates),
                        np.vstack(batch_features),
                        np.concatenate(batch_labels),
                        batch_filenames,
                        np.vstack(batch_raw_color),
                        np.vstack(batch_raw_normals)
                    )
                )
            # TODO WHAT ABOUT POINT2SEGMENT AND SO ON ...
            batch = new_batch
        elif ("train" in self.mode) and self.make_one_pc_noise:
            new_batch = []
            for i in range(0, len(batch), 2):
                if (i + 1) < len(batch):
                    new_batch.append(
                        [
                            np.vstack((batch[i][0], batch[i + 1][0])),
                            np.vstack((batch[i][1], batch[i + 1][1])),
                            np.concatenate(
                                (
                                    batch[i][2],
                                    np.full_like(batch[i + 1][2], self.ignore_label),
                                )
                            ),
                        ]
                    )
                    new_batch.append(
                        [
                            np.vstack((batch[i][0], batch[i + 1][0])),
                            np.vstack((batch[i][1], batch[i + 1][1])),
                            np.concatenate(
                                (
                                    np.full_like(batch[i][2], self.ignore_label),
                                    batch[i + 1][2],
                                )
                            ),
                        ]
                    )
                else:
                    new_batch.append([batch[i][0], batch[i][1], batch[i][2]])
            batch = new_batch
        # return voxelize(batch, self.ignore_label, self.voxel_size, self.probing, self.mode)
        return voxelize(batch, self.ignore_label, self.voxel_size, self.probing, self.mode, task=self.task)


def batch_instances(batch):
    new_batch = []
    for sample in batch:
        for instance_id in np.unique(sample[2][:, 1]):
            new_batch.append(
                (
                    sample[0][sample[2][:, 1] == instance_id],
                    sample[1][sample[2][:, 1] == instance_id],
                    sample[2][sample[2][:, 1] == instance_id][:, 0],
                ),
            )
    return new_batch


def voxelize(batch, ignore_label, voxel_size, probing, mode, task,
             ignore_class_threshold, filter_out_classes, label_offset, num_queries):
    (coordinates, features, labels, original_labels, inverse_maps, original_colors, original_normals,
     original_coordinates, idx) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        []
    )
    voxelization_dict = {
        "ignore_label": ignore_label,
        # "quantization_size": self.voxel_size,
        "return_index": True,
        "return_inverse": True,
    }

    full_res_coords = []

    for sample in batch:
        idx.append(sample[7])
        original_coordinates.append(sample[6])
        original_labels.append(sample[2])
        full_res_coords.append(sample[0])
        original_colors.append(sample[4])
        original_normals.append(sample[5])

        coords = np.floor(sample[0] / voxel_size)
        voxelization_dict.update({"coordinates": torch.from_numpy(coords).to("cpu").contiguous(), "features": sample[1]})

        # maybe this change (_, _, ...) is not necessary and we can directly get out
        # the sample coordinates?
        _, _, unique_map, inverse_map = ME.utils.sparse_quantize(**voxelization_dict)
        inverse_maps.append(inverse_map)

        sample_coordinates = coords[unique_map]
        coordinates.append(torch.from_numpy(sample_coordinates).int())
        sample_features = sample[1][unique_map]
        features.append(torch.from_numpy(sample_features).float())
        if len(sample[2]) > 0:
            sample_labels = sample[2][unique_map]
            labels.append(torch.from_numpy(sample_labels).long())

    # Concatenate all lists
    input_dict = {"coords": coordinates, "feats": features}
    if len(labels) > 0:
        input_dict["labels"] = labels
        coordinates, features, labels = ME.utils.sparse_collate(**input_dict)
    else:
        coordinates, features = ME.utils.sparse_collate(**input_dict)
        labels = torch.Tensor([])

    if probing:
        return (
            NoGpu(coordinates, features, original_labels, inverse_maps, ),
            labels,
        )

    if mode == "test":
        for i in range(len(input_dict["labels"])):
            _, ret_index, ret_inv = np.unique(input_dict["labels"][i][:, 0], return_index=True, return_inverse=True)
            input_dict["labels"][i][:, 0] = torch.from_numpy(ret_inv)
            # input_dict["segment2label"].append(input_dict["labels"][i][ret_index][:, :-1])
    else:
        input_dict["segment2label"] = []

        if "labels" in input_dict:
            for i in range(len(input_dict["labels"])):
                # TODO BIGGER CHANGE CHECK!!!
                _, ret_index, ret_inv = np.unique(input_dict["labels"][i][:, -1], return_index=True, return_inverse=True)
                input_dict["labels"][i][:, -1] = torch.from_numpy(ret_inv)
                input_dict["segment2label"].append(input_dict["labels"][i][ret_index][:, :-1])

    if "labels" in input_dict:
        list_labels = input_dict["labels"]

        target = []
        target_full = []

        if len(list_labels[0].shape) == 1:
            for batch_id in range(len(list_labels)):
                label_ids = list_labels[batch_id].unique()
                if 255 in label_ids:
                    label_ids = label_ids[:-1]

                target.append({
                    'labels': label_ids,
                    'masks': list_labels[batch_id] == label_ids.unsqueeze(1)
                })
        else:
            if mode == "test":
                for i in range(len(input_dict["labels"])):
                    target.append({
                        "point2segment": input_dict["labels"][i][:, 0]
                    })
                    target_full.append({
                        "point2segment": torch.from_numpy(original_labels[i][:, 0]).long()
                    })
            else:
                target = get_instance_masks(list_labels,
                                            list_segments=input_dict["segment2label"],
                                            task=task,
                                            ignore_class_threshold=ignore_class_threshold,
                                            filter_out_classes=filter_out_classes,
                                            label_offset=label_offset)
                for i in range(len(target)):
                    target[i]["point2segment"] = input_dict["labels"][i][:, 2]
                if "train" not in mode:
                    target_full = get_instance_masks([torch.from_numpy(l) for l in original_labels],
                                                     task=task,
                                                     ignore_class_threshold=ignore_class_threshold,
                                                     filter_out_classes=filter_out_classes,
                                                     label_offset=label_offset)
                    for i in range(len(target_full)):
                        target_full[i]["point2segment"] = torch.from_numpy(original_labels[i][:, 2]).long()
    else:
        target = []
        target_full = []
        coordinates = []
        features = []

    if "train" not in mode:
        return (
            NoGpu(coordinates, features, original_labels, inverse_maps, full_res_coords,
                  target_full, original_colors, original_normals, original_coordinates, idx), target,
            [sample[3] for sample in batch]
        )
    else:
        return (
            NoGpu(coordinates, features, original_labels, inverse_maps, full_res_coords), target,
            [sample[3] for sample in batch]
        )


def get_instance_masks(list_labels, task, list_segments=None, ignore_class_threshold=100,
                       filter_out_classes=[], label_offset=0):
    target = []

    for batch_id in range(len(list_labels)):
        label_ids = []
        masks = []
        segment_masks = []
        instance_ids = list_labels[batch_id][:, 1].unique()

        for instance_id in instance_ids:
            if instance_id == -1:
                continue

            # TODO is it possible that a ignore class (255) is an instance???
            # instance == -1 ???
            tmp = list_labels[batch_id][list_labels[batch_id][:, 1] == instance_id]
            label_id = tmp[0, 0]

            if label_id in filter_out_classes:  # floor, wall, undefined==255 is not included
                continue

            if 255 in filter_out_classes and label_id.item() == 255 and tmp.shape[0] < ignore_class_threshold:
                continue

            label_ids.append(label_id)
            masks.append(list_labels[batch_id][:, 1] == instance_id)

            if list_segments:
                segment_mask = torch.zeros(list_segments[batch_id].shape[0]).bool()
                segment_mask[list_labels[batch_id][list_labels[batch_id][:, 1] == instance_id][:, 2].unique()] = True
                segment_masks.append(segment_mask)

        if len(label_ids) == 0:
            return list()

        label_ids = torch.stack(label_ids)
        masks = torch.stack(masks)
        if list_segments:
            segment_masks = torch.stack(segment_masks)

        if task == "semantic_segmentation":
            new_label_ids = []
            new_masks = []
            new_segment_masks = []
            for label_id in label_ids.unique():
                masking = (label_ids == label_id)

                new_label_ids.append(label_id)
                new_masks.append(masks[masking, :].sum(dim=0).bool())

                if list_segments:
                    new_segment_masks.append(segment_masks[masking, :].sum(dim=0).bool())

            label_ids = torch.stack(new_label_ids)
            masks = torch.stack(new_masks)

            if list_segments:
                segment_masks = torch.stack(new_segment_masks)

                target.append({
                    'labels': label_ids,
                    'masks': masks,
                    'segment_mask': segment_masks
                })
            else:
                target.append({
                    'labels': label_ids,
                    'masks': masks
                })
        else:
            l = torch.clamp(label_ids - label_offset, min=0)

            if list_segments:
                target.append({
                    'labels': l,
                    'masks': masks,
                    'segment_mask': segment_masks
                })
            else:
                target.append({
                    'labels': l,
                    'masks': masks
                })
    return target


def make_crops(batch):
    new_batch = []
    # detupling
    for scene in batch:
        new_batch.append([scene[0], scene[1], scene[2]])
    batch = new_batch
    new_batch = []
    for scene in batch:
        # move to center for better quadrant split
        scene[0][:, :3] -= scene[0][:, :3].mean(0)

        # BUGFIX - there always would be a point in every quadrant
        scene[0] = np.vstack(
            (
                scene[0],
                np.array(
                    [
                        [0.1, 0.1, 0.1],
                        [0.1, -0.1, 0.1],
                        [-0.1, 0.1, 0.1],
                        [-0.1, -0.1, 0.1],
                    ]
                ),
            )
        )
        scene[1] = np.vstack((scene[1], np.zeros((4, scene[1].shape[1]))))
        scene[2] = np.concatenate((scene[2], np.full_like((scene[2]), 255)[:4]))

        crop = scene[0][:, 0] > 0
        crop &= scene[0][:, 1] > 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] > 0
        crop &= scene[0][:, 1] < 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] < 0
        crop &= scene[0][:, 1] > 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] < 0
        crop &= scene[0][:, 1] < 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

    # moving all of them to center
    for i in range(len(new_batch)):
        new_batch[i][0][:, :3] -= new_batch[i][0][:, :3].mean(0)
    return new_batch


class NoGpu:
    def __init__(
            self, coordinates, features, original_labels=None, inverse_maps=None, full_res_coords=None,
            target_full=None, original_colors=None, original_normals=None, original_coordinates=None,
            idx=None
    ):
        """ helper class to prevent gpu loading on lightning """
        self.coordinates = coordinates
        self.features = features
        self.original_labels = original_labels
        self.inverse_maps = inverse_maps
        self.full_res_coords = full_res_coords
        self.target_full = target_full
        self.original_colors = original_colors
        self.original_normals = original_normals
        self.original_coordinates = original_coordinates
        self.idx = idx


class NoGpuMask:
    def __init__(
            self, coordinates, features, original_labels=None, inverse_maps=None, masks=None, labels=None
    ):
        """ helper class to prevent gpu loading on lightning """
        self.coordinates = coordinates
        self.features = features
        self.original_labels = original_labels
        self.inverse_maps = inverse_maps

        self.masks = masks
        self.labels = labels