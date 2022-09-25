# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch


def check_aspect(crop_range, aspect_min):
    xy_aspect = np.min(crop_range[:2]) / np.max(crop_range[:2])
    xz_aspect = np.min(crop_range[[0, 2]]) / np.max(crop_range[[0, 2]])
    yz_aspect = np.min(crop_range[1:]) / np.max(crop_range[1:])
    return (
        (xy_aspect >= aspect_min)
        or (xz_aspect >= aspect_min)
        or (yz_aspect >= aspect_min)
    )


class RandomCuboid(object):
    """
    RandomCuboid augmentation from DepthContrast [https://arxiv.org/abs/2101.02691]
    We slightly modify this operation to account for object detection.
    This augmentation randomly crops a cuboid from the input and
    ensures that the cropped cuboid contains at least one bounding box
    """

    def __init__(
        self,
        min_points,
        #aspect=0.8,
        crop_length=6.0,
        version1=True
    ):
        #self.aspect = aspect
        self.crop_length = crop_length
        self.min_points = min_points
        self.version1 = version1

    def __call__(self, point_cloud):
        if point_cloud.shape[0] < self.min_points:
            print("too small pcd")
            return np.ones(point_cloud.shape[0], dtype=np.bool)

        range_xyz = np.max(point_cloud[:, :2], axis=0) - np.min(
            point_cloud[:, :2], axis=0
        )

        for _ in range(100):
            #crop_range = self.min_crop + np.random.rand(3) * (
            #    self.max_crop - self.min_crop
            #)
            #crop_range[-1] = 999.
            # if not check_aspect(crop_range, self.aspect):
            #     continue

            sample_center = point_cloud[:, :2].min(axis=0) + range_xyz/2

            if self.version1:
                offset_x = np.random.uniform(-range_xyz[0]/4,range_xyz[0]/4)
                offset_y = np.random.uniform(-range_xyz[1]/4,range_xyz[1]/4)
            else:
                offset_x = np.random.uniform(-(range_xyz[0]/2) + self.crop_length / 4,
                                             +(range_xyz[0]/2) - self.crop_length / 4)
                offset_y = np.random.uniform(-(range_xyz[1]/2) + self.crop_length / 4,
                                             +(range_xyz[1]/2) - self.crop_length / 4)

            sample_center[0] = sample_center[0] + offset_x
            sample_center[1] = sample_center[1] + offset_y

            min_xy = sample_center - self.crop_length / 2
            max_xy = sample_center + self.crop_length / 2

            upper_idx = (
                np.sum((point_cloud[:, :2] <= max_xy).astype(np.int32), 1) == 2
            )
            lower_idx = (
                np.sum((point_cloud[:, :2] >= min_xy).astype(np.int32), 1) == 2
            )

            new_pointidx = (upper_idx) & (lower_idx)

            if np.sum(new_pointidx) < self.min_points:
                print("TOO SMALL")
                continue

            return new_pointidx

        # fallback
        print("FALLBACK")
        return np.ones(point_cloud.shape[0], dtype=np.bool)
