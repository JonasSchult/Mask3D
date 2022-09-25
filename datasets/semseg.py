import logging
from itertools import product
from pathlib import Path
from random import random, sample, uniform
from typing import List, Optional, Tuple, Union
from random import choice
from copy import deepcopy
import torch
from datasets.random_cuboid import RandomCuboid

import albumentations as A
import numpy as np
import scipy
import volumentations as V
import yaml
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SemanticSegmentationDataset(Dataset):
    """Docstring for SemanticSegmentationDataset. """

    def __init__(
        self,
        data_dir: Optional[Union[str, Tuple[str]]] = "data/processed/scannet",
        label_db_filepath: Optional[
            str
        ] = "configs/scannet_preprocessing/label_database.yaml",
        # mean std values from scannet
        color_mean_std: Optional[Union[str, Tuple[Tuple[float]]]] = (
            (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
            (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
        ),
        mode: Optional[str] = "train",
        add_colors: Optional[bool] = True,
        add_normals: Optional[bool] = True,
        add_raw_coordinates: Optional[bool] = False,
        add_instance: Optional[bool] = False,
        num_labels: Optional[int] = -1,
        data_percent: Optional[float] = 1.0,
        ignore_label: Optional[Union[int, Tuple[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        image_augmentations_path: Optional[str] = None,
        instance_oversampling=0,
        place_around_existing=False,
        max_cut_region=0,
        point_per_cut=100,
        flip_in_center=False,
        noise_rate=0.0,
        resample_points=0.0,
        cache_data=False,
        add_unlabeled_pc=False,
        task="instance_segmentation",
        cropping=False,
        cropping_args=None,
        is_tta=False,
        crop_min_size=20000,
        crop_length=6.0,
        cropping_v1=True,
        reps_per_epoch=1,
        area=-1,
        on_crops=False
    ):
        assert task in ["instance_segmentation", "semantic_segmentation"], "unknown task"
        self.task = task

        self.area = area

        self.reps_per_epoch = reps_per_epoch

        self.cropping = cropping
        self.cropping_args = cropping_args
        self.is_tta = is_tta
        self.on_crops = on_crops

        self.crop_min_size = crop_min_size
        self.crop_length = crop_length

        self.version1 = cropping_v1

        self.random_cuboid = RandomCuboid(self.crop_min_size,
                                          crop_length=self.crop_length,
                                          version1=self.version1)

        self.mode = mode
        self.data_dir = data_dir
        self.add_unlabeled_pc = add_unlabeled_pc
        if add_unlabeled_pc:
            self.other_database = self._load_yaml(
                Path(data_dir).parent / "matterport" / "train_database.yaml"
            )
        if type(data_dir) == str:
            self.data_dir = [self.data_dir]
        self.ignore_label = ignore_label
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.add_instance = add_instance
        self.add_raw_coordinates = add_raw_coordinates
        self.instance_oversampling = instance_oversampling
        self.place_around_existing = place_around_existing
        self.max_cut_region = max_cut_region
        self.point_per_cut = point_per_cut
        self.flip_in_center = flip_in_center
        self.noise_rate = noise_rate
        self.resample_points = resample_points

        # loading database files
        self._data = []
        for database_path in self.data_dir:
            database_path = Path(database_path)
            if "s3dis" not in str(database_path):
                if not (database_path / f"{mode}_database.yaml").exists():
                    print(f"generate {database_path}/{mode}_database.yaml first")
                    exit()
                self._data.extend(self._load_yaml(database_path / f"{mode}_database.yaml"))
            else:
                mode_s3dis = f"Area_{self.area}"
                if self.mode == "train":
                    mode_s3dis = "train_" + mode_s3dis
                if not (database_path / f"{mode_s3dis}_database.yaml").exists():
                    print(f"generate {database_path}/{mode_s3dis}_database.yaml first")
                    exit()
                self._data.extend(self._load_yaml(database_path / f"{mode_s3dis}_database.yaml"))
        if data_percent < 1.0:
            self._data = sample(self._data, int(len(self._data) * data_percent))
        labels = self._load_yaml(Path(label_db_filepath))

        # if working only on classes for validation - discard others
        self._labels = self._select_correct_labels(labels, num_labels)

        if instance_oversampling > 0:
            self.instance_data = self._load_yaml(
                Path(label_db_filepath).parent / "instance_database.yaml"
            )

        # normalize color channels
        if "s3dis" in str(self.data_dir[0]):
            color_mean_std = color_mean_std.replace("color_mean_std.yaml", f"Area_{self.area}_color_mean_std.yaml")

        if Path(str(color_mean_std)).exists():
            color_mean_std = self._load_yaml(color_mean_std)
            color_mean, color_std = (
                tuple(color_mean_std["mean"]),
                tuple(color_mean_std["std"]),
            )
        elif len(color_mean_std[0]) == 3 and len(color_mean_std[1]) == 3:
            color_mean, color_std = color_mean_std[0], color_mean_std[1]
        else:
            logger.error("pass mean and std as tuple of tuples, or as an .yaml file")

        # augmentations
        self.volume_augmentations = V.NoOp()
        if (volume_augmentations_path is not None) and (
            volume_augmentations_path != "none"
        ):
            self.volume_augmentations = V.load(
                Path(volume_augmentations_path), data_format="yaml"
            )
        self.image_augmentations = A.NoOp()
        if (image_augmentations_path is not None) and (
            image_augmentations_path != "none"
        ):
            self.image_augmentations = A.load(
                Path(image_augmentations_path), data_format="yaml"
            )
        # mandatory color augmentation
        if add_colors:
            self.normalize_color = A.Normalize(mean=color_mean, std=color_std)

    @property
    def color_map(self):
        return torch.FloatTensor(
            [[255, 255, 255],  # unlabeled
             [174, 199, 232],  # wall
                [152, 223, 138],  # floor
                [31, 119, 180],  # cabinet
                [255, 187, 120],  # bed
                [188, 189, 34],  # chair
                [140, 86, 75],   # sofa
                [255, 152, 150],  # table
                [214, 39, 40],   # door
                [197, 176, 213],  # window
                [148, 103, 189],  # bookshelf
                [196, 156, 148],  # picture
                [23, 190, 207],  # counter
                [247, 182, 210],  # desk
                [219, 219, 141],  # curtain
                [255, 127, 14],  # refrigerator
                [158, 218, 229],  # shower curtain
                [44, 160, 44],   # toilet
                [112, 128, 144],  # sink
                [227, 119, 194],  # bathtub
                [82, 84, 163]])  # otherfurn

    def __len__(self):
        if self.is_tta:
            return 5*len(self.data)
        else:
            return self.reps_per_epoch*len(self.data)

    def __getitem__(self, idx: int):
        idx = idx % len(self.data)
        if self.is_tta:
            idx = idx % len(self.data)

        points = np.load(self.data[idx]["filepath"].replace("../../", ""))

        if "train" in self.mode and any([x in self.data_dir[0].split("/")[-1] for x in ["s3dis"]]):
            inds = self.random_cuboid(points)
            points = points[inds]

        coordinates, color, normals, segments, labels = (
            points[:, :3],
            points[:, 3:6],
            points[:, 6:9],
            points[:, 9],
            points[:, 10:],
        )

        raw_coordinates = coordinates.copy()
        raw_color = color
        raw_normals = normals

        if not self.add_colors:
            color = np.ones((len(color), 3))

        # volume and image augmentations for train
        if "train" in self.mode or self.is_tta:
            if self.cropping:
                new_idx = self.random_cuboid(coordinates, labels[:,1],
                                             self._remap_from_zero(labels[:, 0].copy()))

                coordinates = coordinates[new_idx]
                color = color[new_idx]
                labels = labels[new_idx]
                segments = segments[new_idx]
                raw_color = raw_color[new_idx]
                raw_normals = raw_normals[new_idx]
                normals = normals[new_idx]
                points = points[new_idx]


            coordinates -= coordinates.mean(0)
            coordinates += np.random.uniform(coordinates.min(0), coordinates.max(0)) / 2

            if self.instance_oversampling > 0.0:
                coordinates, color, normals, labels = self.augment_individual_instance(
                    coordinates, color, normals, labels, self.instance_oversampling
                )

            if self.flip_in_center:
                coordinates = flip_in_center(coordinates)

            for i in (0, 1):
                if random() < 0.5:
                    coord_max = np.max(points[:, i])
                    coordinates[:, i] = coord_max - coordinates[:, i]
            if random() < 0.95:
                for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
                    coordinates = elastic_distortion(
                        coordinates, granularity, magnitude
                    )
            aug = self.volume_augmentations(
                points=coordinates, normals=normals, features=color, labels=labels,
            )
            coordinates, color, normals, labels = (
                aug["points"],
                aug["features"],
                aug["normals"],
                aug["labels"],
            )
            pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
            color = np.squeeze(self.image_augmentations(image=pseudo_image)["image"])

            if self.point_per_cut != 0:
                number_of_cuts = int(len(coordinates) / self.point_per_cut)
                for _ in range(number_of_cuts):
                    size_of_cut = np.random.uniform(0.05, self.max_cut_region)
                    # not wall, floor or empty
                    point = choice(coordinates)
                    x_min = point[0] - size_of_cut
                    x_max = x_min + size_of_cut
                    y_min = point[1] - size_of_cut
                    y_max = y_min + size_of_cut
                    z_min = point[2] - size_of_cut
                    z_max = z_min + size_of_cut
                    indexes = crop(
                        coordinates, x_min, y_min, z_min, x_max, y_max, z_max
                    )
                    coordinates, normals, color, labels = (
                        coordinates[~indexes],
                        normals[~indexes],
                        color[~indexes],
                        labels[~indexes],
                    )

            # if self.noise_rate > 0:
            #     coordinates, color, normals, labels = random_points(
            #         coordinates,
            #         color,
            #         normals,
            #         labels,
            #         self.noise_rate,
            #         self.ignore_label,
            #     )

            if (self.resample_points > 0) or (self.noise_rate > 0):
                coordinates, color, normals, labels = random_around_points(
                    coordinates,
                    color,
                    normals,
                    labels,
                    self.resample_points,
                    self.noise_rate,
                    self.ignore_label,
                )

            if self.add_unlabeled_pc:
                if random() < 0.8:
                    new_points = np.load(
                        self.other_database[
                            np.random.randint(0, len(self.other_database) - 1)
                        ]["filepath"]
                    )
                    (
                        unlabeled_coords,
                        unlabeled_color,
                        unlabeled_normals,
                        unlabeled_labels,
                    ) = (
                        new_points[:, :3],
                        new_points[:, 3:6],
                        new_points[:, 6:9],
                        new_points[:, 9:],
                    )
                    unlabeled_coords -= unlabeled_coords.mean(0)
                    unlabeled_coords += (
                        np.random.uniform(
                            unlabeled_coords.min(0), unlabeled_coords.max(0)
                        )
                        / 2
                    )

                    aug = self.volume_augmentations(
                        points=unlabeled_coords,
                        normals=unlabeled_normals,
                        features=unlabeled_color,
                        labels=unlabeled_labels,
                    )
                    (
                        unlabeled_coords,
                        unlabeled_color,
                        unlabeled_normals,
                        unlabeled_labels,
                    ) = (
                        aug["points"],
                        aug["features"],
                        aug["normals"],
                        aug["labels"],
                    )
                    pseudo_image = unlabeled_color.astype(np.uint8)[np.newaxis, :, :]
                    unlabeled_color = np.squeeze(
                        self.image_augmentations(image=pseudo_image)["image"]
                    )

                    coordinates = np.concatenate((coordinates, unlabeled_coords))
                    color = np.concatenate((color, unlabeled_color))
                    normals = np.concatenate((normals, unlabeled_normals))
                    labels = np.concatenate(
                        (labels, np.full_like(unlabeled_labels, self.ignore_label))
                    )

        # normalize color information
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.normalize_color(image=pseudo_image)["image"])

        # prepare labels and map from 0 to 20(40)
        labels = labels.astype(np.int32)
        if labels.size > 0:
            labels[:, 0] = self._remap_from_zero(labels[:, 0])
            if not self.add_instance:
                # taking only first column, which is segmentation label, not instance
                labels = labels[:, 0].flatten()[..., None]

        labels = np.hstack((labels, segments[..., None].astype(np.int32)))

        features = color
        if self.add_normals:
            features = np.hstack((features, normals))
        if self.add_raw_coordinates:
            features = np.hstack((features, coordinates))

        if self.data[idx]['raw_filepath'].split("/")[-2] in ['scene0636_00', 'scene0154_00']:
            return self.__getitem__(0)

        if "s3dis" in self.data_dir[0].split("/")[-1]:
            return coordinates, features, labels, self.data[idx]['area'] + "_" + self.data[idx]['scene'], raw_color, raw_normals, raw_coordinates, idx
        else:
            return coordinates, features, labels, self.data[idx]['raw_filepath'].split("/")[-2], raw_color, raw_normals, raw_coordinates, idx

    @property
    def data(self):
        """ database file containing information about preproscessed dataset """
        return self._data

    @property
    def label_info(self):
        """ database file containing information labels used by dataset """
        return self._labels

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            file = yaml.safe_load(f)
        return file

    def _select_correct_labels(self, labels, num_labels):
        number_of_validation_labels = 0
        number_of_all_labels = 0
        for k, v, in labels.items():
            number_of_all_labels += 1
            if v["validation"]:
                number_of_validation_labels += 1

        if num_labels == number_of_all_labels:
            return labels
        elif num_labels == number_of_validation_labels:
            valid_labels = dict()
            for k, v, in labels.items():
                if v["validation"]:
                    valid_labels.update({k: v})
            return valid_labels
        else:
            msg = f"""not available number labels, select from:
            {number_of_validation_labels}, {number_of_all_labels}"""
            raise ValueError(msg)

    def _remap_from_zero(self, labels):
        labels[~np.isin(labels, list(self.label_info.keys()))] = self.ignore_label
        # remap to the range from 0
        for i, k in enumerate(self.label_info.keys()):
            labels[labels == k] = i
        return labels

    def _remap_model_output(self, output):
        output = np.array(output)
        output_remapped = output.copy()
        for i, k in enumerate(self.label_info.keys()):
            output_remapped[output == i] = k
        return output_remapped

    def augment_individual_instance(
        self, coordinates, color, normals, labels, oversampling=1.0
    ):
        max_instance = int(len(np.unique(labels[:, 1])))
        # randomly selecting half of non-zero instances
        for instance in range(0, int(max_instance * oversampling)):
            if self.place_around_existing:
                center = choice(
                    coordinates[labels[:, 1] == choice(np.unique(labels[:, 1]))]
                )
            else:
                center = np.array([uniform(-5, 5), uniform(-5, 5), uniform(-0.5, 2)])
            instance = choice(choice(self.instance_data))
            instance = np.load(instance["instance_filepath"])
            # centering two objects
            instance[:, :3] = instance[:, :3] - instance[:, :3].mean(axis=0) + center
            max_instance = max_instance + 1
            instance[:, -1] = max_instance
            aug = V.Compose(
                [
                    V.Scale3d(),
                    V.RotateAroundAxis3d(rotation_limit=np.pi / 24, axis=(1, 0, 0)),
                    V.RotateAroundAxis3d(rotation_limit=np.pi / 24, axis=(0, 1, 0)),
                    V.RotateAroundAxis3d(rotation_limit=np.pi, axis=(0, 0, 1)),
                ]
            )(
                points=instance[:, :3],
                features=instance[:, 3:6],
                normals=instance[:, 6:9],
                labels=instance[:, 9:],
            )
            coordinates = np.concatenate((coordinates, aug["points"]))
            color = np.concatenate((color, aug["features"]))
            normals = np.concatenate((normals, aug["normals"]))
            labels = np.concatenate((labels, aug["labels"]))

        return coordinates, color, normals, labels


def elastic_distortion(pointcloud, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space.

    pointcloud: numpy array of (number of points, at least 3 spatial dims)
    granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: noise multiplier
  """
    blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
    blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
    blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
    coords = pointcloud[:, :3]
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
        noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(
            coords_min - granularity,
            coords_min + granularity * (noise_dim - 2),
            noise_dim,
        )
    ]
    interp = scipy.interpolate.RegularGridInterpolator(
        ax, noise, bounds_error=0, fill_value=0
    )
    pointcloud[:, :3] = coords + interp(coords) * magnitude
    return pointcloud


def crop(points, x_min, y_min, z_min, x_max, y_max, z_max):
    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max and z_min < z_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, z_min = {z_min},"
            " x_max = {x_max}, y_max = {y_max}, z_max = {z_max})".format(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                z_min=z_min,
                z_max=z_max,
            )
        )
    inds = np.all(
        [
            (points[:, 0] >= x_min),
            (points[:, 0] < x_max),
            (points[:, 1] >= y_min),
            (points[:, 1] < y_max),
            (points[:, 2] >= z_min),
            (points[:, 2] < z_max),
        ],
        axis=0,
    )
    return inds


def flip_in_center(coordinates):
    # moving coordinates to center
    coordinates -= coordinates.mean(0)
    aug = V.Compose(
        [
            V.Flip3d(axis=(0, 1, 0), always_apply=True),
            V.Flip3d(axis=(1, 0, 0), always_apply=True),
        ]
    )

    first_crop = coordinates[:, 0] > 0
    first_crop &= coordinates[:, 1] > 0
    # x -y
    second_crop = coordinates[:, 0] > 0
    second_crop &= coordinates[:, 1] < 0
    # -x y
    third_crop = coordinates[:, 0] < 0
    third_crop &= coordinates[:, 1] > 0
    # -x -y
    fourth_crop = coordinates[:, 0] < 0
    fourth_crop &= coordinates[:, 1] < 0

    if first_crop.size > 1:
        coordinates[first_crop] = aug(points=coordinates[first_crop])["points"]
    if second_crop.size > 1:
        minimum = coordinates[second_crop].min(0)
        minimum[2] = 0
        minimum[0] = 0
        coordinates[second_crop] = aug(points=coordinates[second_crop])["points"]
        coordinates[second_crop] += minimum
    if third_crop.size > 1:
        minimum = coordinates[third_crop].min(0)
        minimum[2] = 0
        minimum[1] = 0
        coordinates[third_crop] = aug(points=coordinates[third_crop])["points"]
        coordinates[third_crop] += minimum
    if fourth_crop.size > 1:
        minimum = coordinates[fourth_crop].min(0)
        minimum[2] = 0
        coordinates[fourth_crop] = aug(points=coordinates[fourth_crop])["points"]
        coordinates[fourth_crop] += minimum

    return coordinates


def random_around_points(
    coordinates, color, normals, labels, rate=0.2, noise_rate=0, ignore_label=255
):
    coord_indexes = sample(
        list(range(len(coordinates))), k=int(len(coordinates) * rate)
    )
    noisy_coordinates = deepcopy(coordinates[coord_indexes])
    noisy_coordinates += np.random.uniform(
        -0.2 - noise_rate, 0.2 + noise_rate, size=noisy_coordinates.shape
    )

    if noise_rate > 0:
        noisy_color = np.random.randint(0, 255, size=noisy_coordinates.shape)
        noisy_normals = np.random.rand(*noisy_coordinates.shape) * 2 - 1
        noisy_labels = np.full(labels[coord_indexes].shape, ignore_label)

        coordinates = np.vstack((coordinates, noisy_coordinates))
        color = np.vstack((color, noisy_color))
        normals = np.vstack((normals, noisy_normals))
        labels = np.vstack((labels, noisy_labels))
    else:
        noisy_color = deepcopy(color[coord_indexes])
        noisy_normals = deepcopy(normals[coord_indexes])
        noisy_labels = deepcopy(labels[coord_indexes])

        coordinates = np.vstack((coordinates, noisy_coordinates))
        color = np.vstack((color, noisy_color))
        normals = np.vstack((normals, noisy_normals))
        labels = np.vstack((labels, noisy_labels))

    return coordinates, color, normals, labels


def random_points(
    coordinates, color, normals, labels, noise_rate=0.6, ignore_label=255
):
    max_boundary = coordinates.max(0) + 0.1
    min_boundary = coordinates.min(0) - 0.1

    noisy_coordinates = int((max(max_boundary) - min(min_boundary)) / noise_rate)

    noisy_coordinates = np.array(
        list(
            product(
                np.linspace(min_boundary[0], max_boundary[0], noisy_coordinates),
                np.linspace(min_boundary[1], max_boundary[1], noisy_coordinates),
                np.linspace(min_boundary[2], max_boundary[2], noisy_coordinates),
            )
        )
    )
    noisy_coordinates += np.random.uniform(
        -noise_rate, noise_rate, size=noisy_coordinates.shape
    )

    noisy_color = np.random.randint(0, 255, size=noisy_coordinates.shape)
    noisy_normals = np.random.rand(*noisy_coordinates.shape) * 2 - 1
    noisy_labels = np.full((noisy_coordinates.shape[0], labels.shape[1]), ignore_label)

    coordinates = np.vstack((coordinates, noisy_coordinates))
    color = np.vstack((color, noisy_color))
    normals = np.vstack((normals, noisy_normals))
    labels = np.vstack((labels, noisy_labels))
    return coordinates, color, normals, labels
