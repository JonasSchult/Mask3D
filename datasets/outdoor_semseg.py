import logging
from pathlib import Path
from typing import List, Optional, Union, Tuple
from random import random

import numpy as np
import volumentations as V
import yaml
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LidarDataset(Dataset):
    def __init__(
        self,
        data_dir: Optional[Union[str, Tuple[str]]] = "data/processed/semantic_kitti",
        label_db_filepath: Optional[
            str
        ] = "./data/processed/semantic_kitti/label_database.yaml",
        mode: Optional[str] = "train",
        add_reflection: Optional[bool] = True,
        add_distance: Optional[bool] = False,
        add_instance: Optional[bool] = True,
        num_labels: Optional[int] = -1,
        data_percent: Optional[float] = 1.0,
        ignore_label: Optional[Union[int, List[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        sweep: Optional[int] = 1,
    ):
        self.mode = mode
        self.data_dir = data_dir
        if type(data_dir) == str:
            self.data_dir = [self.data_dir]
        self.ignore_label = ignore_label
        self.add_instance = add_instance
        self.add_distance = add_distance
        self.add_reflection = add_reflection

        # loading database files
        self._data = []
        for database_path in self.data_dir:
            database_path = Path(database_path)
            if not (database_path / f"{mode}_database.yaml").exists():
                print(f"generate {database_path}/{mode}_database.yaml first")
                exit()
            self._data.extend(self._load_yaml(database_path / f"{mode}_database.yaml"))

        labels = self._load_yaml(Path(label_db_filepath))
        self._labels = self._select_correct_labels(labels, num_labels)

        # augmentations
        self.volume_augmentations = V.NoOp()
        if volume_augmentations_path is not None:
            self.volume_augmentations = V.load(
                volume_augmentations_path, data_format="yaml"
            )

        # reformulating in sweeps
        data = [[]]
        last_scene = self._data[0]["scene"]
        for x in self._data:
            if x["scene"] == last_scene:
                data[-1].append(x)
            else:
                last_scene = x["scene"]
                data.append([x])
        for i in range(len(data)):
            data[i] = list(self.chunks(data[i], sweep))
        self._data = [val for sublist in data for val in sublist]

        if data_percent < 1.0:
            self._data = self._data[: int(len(self._data) * data_percent)]

    @staticmethod
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        points = []
        for sweep in self.data[idx]:
            points.append(np.load(sweep["filepath"]))
            # rotate
            points[-1][:, :3] = points[-1][:, :3] @ np.array(sweep["pose"])[:3, :3]
            # translate
            points[-1][:, :3] += np.array(sweep["pose"])[:3, 3]
        points = np.vstack(points)

        coordinates, features, labels = (
            points[:, :3],
            points[:, 3:-2],
            points[:, -2:],
        )

        if not self.add_reflection:
            features = np.ones(np.ones((len(coordinates), 1)))

        if self.add_distance:
            center_coordinate = coordinates.mean(0)
            features = np.hstack(
                (
                    features,
                    np.linalg.norm(coordinates - center_coordinate, axis=1)[
                        :, np.newaxis
                    ],
                )
            )

        # volume and image augmentations for train
        if "train" in self.mode:
            coordinates -= coordinates.mean(0)
            if 0.5 > random():
                coordinates += (
                    np.random.uniform(coordinates.min(0), coordinates.max(0)) / 2
                )
            aug = self.volume_augmentations(
                points=coordinates, features=features, labels=labels,
            )
            coordinates, features, labels = (
                aug["points"],
                aug["features"],
                aug["labels"],
            )

        # prepare labels and map from 0 to 20(40)
        labels = labels.astype(np.int32)
        if labels.size > 0:
            labels[:, 0] = self._remap_from_zero(labels[:, 0])
            if not self.add_instance:
                # taking only first column, which is segmentation label, not instance
                labels = labels[:, 0].flatten()

        return coordinates, features, labels

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
