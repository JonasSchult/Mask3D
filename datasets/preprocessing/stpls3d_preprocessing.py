import re
import os
import numpy as np
from fire import Fire
from natsort import natsorted
from loguru import logger
import pandas as pd

from datasets.preprocessing.base_preprocessing import BasePreprocessing


class STPLS3DPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir: str = "../../data/raw/stpls3d",
        save_dir: str = "../../data/processed/stpls3d",
        modes: tuple = ("train", "validation", "test"),
        n_jobs: int = -1,
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)

        # https://github.com/meidachen/STPLS3D/blob/main/HAIS/STPLS3DInstanceSegmentationChallenge_Codalab_Evaluate.py#L31
        CLASS_LABELS = [
            "Build",
            "LowVeg",
            "MediumVeg",
            "HighVeg",
            "Vehicle",
            "Truck",
            "Aircraft",
            "MilitaryVeh",
            "Bike",
            "Motorcycle",
            "LightPole",
            "StreetSign",
            "Clutter",
            "Fence",
        ]
        VALID_CLASS_IDS = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        )

        self.class_map = {
            "Ground": 0,
            "Build": 1,
            "LowVeg": 2,
            "MediumVeg": 3,
            "HighVeg": 4,
            "Vehicle": 5,
            "Truck": 6,
            "Aircraft": 7,
            "MilitaryVeh": 8,
            "Bike": 9,
            "Motorcycle": 10,
            "LightPole": 11,
            "StreetSign": 12,
            "Clutter": 13,
            "Fence": 14,
        }

        self.color_map = [
            [0, 255, 0],  # Ground
            [0, 0, 255],  # Build
            [0, 255, 255],  # LowVeg
            [255, 255, 0],  # MediumVeg
            [255, 0, 255],  # HiVeg
            [100, 100, 255],  # Vehicle
            [200, 200, 100],  # Truck
            [170, 120, 200],  # Aircraft
            [255, 0, 0],  # MilitaryVec
            [200, 100, 100],  # Bike
            [10, 200, 100],  # Motorcycle
            [200, 200, 200],  # LightPole
            [50, 50, 50],  # StreetSign
            [60, 130, 60],  # Clutter
            [130, 30, 60],
        ]  # Fence

        self.create_label_database()

        for mode in self.modes:
            filepaths = []
            for scene_path in [
                f.path for f in os.scandir(self.data_dir / mode)
            ]:
                filepaths.append(scene_path)
            self.files[mode] = natsorted(filepaths)

    def create_label_database(self):
        label_database = dict()
        for class_name, class_id in self.class_map.items():
            label_database[class_id] = {
                "color": self.color_map[class_id],
                "name": class_name,
                "validation": True,
            }

        self._save_yaml(self.save_dir / "label_database.yaml", label_database)
        return label_database

    def process_file(self, filepath, mode):
        """process_file.

        Please note, that for obtaining segmentation labels ply files were used.

        Args:
            filepath: path to the main ply file
            mode: train, test or validation

        Returns:
            filebase: info about file
        """
        filebase = {
            "filepath": filepath,
            "scene": filepath.split("/")[-1],
            "raw_filepath": str(filepath),
            "file_len": -1,
        }

        points = pd.read_csv(filepath, header=None).values

        filebase["raw_segmentation_filepath"] = ""

        # add segment id as additional feature (DUMMY)
        if mode in ["train", "validation"]:
            points = np.hstack(
                (
                    points,
                    np.ones(points.shape[0])[..., None],  # normal 1
                    np.ones(points.shape[0])[..., None],  # normal 2
                    np.ones(points.shape[0])[..., None],  # normal 3
                    np.ones(points.shape[0])[..., None],
                )
            )  # segments
        else:
            # we need to add dummies for semantics and instances
            points = np.hstack(
                (
                    points,
                    np.ones(points.shape[0])[..., None],  # semantic class
                    np.ones(points.shape[0])[..., None],  # instance id
                    np.ones(points.shape[0])[..., None],  # normal 1
                    np.ones(points.shape[0])[..., None],  # normal 2
                    np.ones(points.shape[0])[..., None],  # normal 3
                    np.ones(points.shape[0])[..., None],
                )
            )  # segments

        points = points[
            :, [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 6, 7]
        ]  # move segments after RGB

        # move point clouds to be in positive range (important for split pointcloud function)
        points[:, :3] = points[:, :3] - points[:, :3].min(0)

        points = points.astype(np.float32)

        if mode == "test":
            points = points[:, :-2]
        else:
            points[
                points[:, -1] == -100.0, -1
            ] = -1  # -1 indicates "no instance"

        file_len = len(points)
        filebase["file_len"] = file_len

        processed_filepath = (
            self.save_dir
            / mode
            / f"{filebase['scene'].replace('.txt', '')}.npy"
        )
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        if mode in ["validation", "test"]:
            blocks = self.splitPointCloud(points)

            filebase["instance_gt_filepath"] = []
            filebase["filepath_crop"] = []
            for block_id, block in enumerate(blocks):
                if len(block) > 10000:
                    if mode == "validation":
                        new_instance_ids = np.unique(
                            block[:, -1], return_inverse=True
                        )[1]

                        assert new_instance_ids.shape[0] == block.shape[0]
                        # == 0 means -1 == no instance
                        # new_instance_ids[new_instance_ids == 0]
                        assert (
                            new_instance_ids.max() < 1000
                        ), "we cannot encode when there are more than 999 instances in a block"

                        gt_data = (block[:, -2]) * 1000 + new_instance_ids

                        processed_gt_filepath = (
                            self.save_dir
                            / "instance_gt"
                            / mode
                            / f"{filebase['scene'].replace('.txt', '')}_{block_id}.txt"
                        )
                        if not processed_gt_filepath.parent.exists():
                            processed_gt_filepath.parent.mkdir(
                                parents=True, exist_ok=True
                            )
                        np.savetxt(
                            processed_gt_filepath,
                            gt_data.astype(np.int32),
                            fmt="%d",
                        )
                        filebase["instance_gt_filepath"].append(
                            str(processed_gt_filepath)
                        )

                    processed_filepath = (
                        self.save_dir
                        / mode
                        / f"{filebase['scene'].replace('.txt', '')}_{block_id}.npy"
                    )
                    if not processed_filepath.parent.exists():
                        processed_filepath.parent.mkdir(
                            parents=True, exist_ok=True
                        )
                    np.save(processed_filepath, block.astype(np.float32))
                    filebase["filepath_crop"].append(str(processed_filepath))
                else:
                    print("block was smaller than 1000 points")
                    assert False

        filebase["color_mean"] = [
            float((points[:, 3] / 255).mean()),
            float((points[:, 4] / 255).mean()),
            float((points[:, 5] / 255).mean()),
        ]
        filebase["color_std"] = [
            float(((points[:, 3] / 255) ** 2).mean()),
            float(((points[:, 4] / 255) ** 2).mean()),
            float(((points[:, 5] / 255) ** 2).mean()),
        ]
        return filebase

    def compute_color_mean_std(
        self,
        train_database_path: str = "./data/processed/stpls3d/train_database.yaml",
    ):
        train_database = self._load_yaml(train_database_path)
        color_mean, color_std = [], []
        for sample in train_database:
            color_std.append(sample["color_std"])
            color_mean.append(sample["color_mean"])

        color_mean = np.array(color_mean).mean(axis=0)
        color_std = np.sqrt(np.array(color_std).mean(axis=0) - color_mean**2)
        feats_mean_std = {
            "mean": [float(each) for each in color_mean],
            "std": [float(each) for each in color_std],
        }
        self._save_yaml(self.save_dir / "color_mean_std.yaml", feats_mean_std)

    def splitPointCloud(self, cloud, size=50.0, stride=50):
        limitMax = np.amax(cloud[:, 0:3], axis=0)
        width = int(np.ceil((limitMax[0] - size) / stride)) + 1
        depth = int(np.ceil((limitMax[1] - size) / stride)) + 1
        cells = [
            (x * stride, y * stride)
            for x in range(width)
            for y in range(depth)
        ]
        blocks = []
        for (x, y) in cells:
            xcond = (cloud[:, 0] <= x + size) & (cloud[:, 0] >= x)
            ycond = (cloud[:, 1] <= y + size) & (cloud[:, 1] >= y)
            cond = xcond & ycond
            block = cloud[cond, :]
            blocks.append(block)
        return blocks

    @logger.catch
    def fix_bugs_in_labels(self):
        pass

    def _parse_scene_subscene(self, name):
        scene_match = re.match(r"scene(\d{4})_(\d{2})", name)
        return int(scene_match.group(1)), int(scene_match.group(2))


if __name__ == "__main__":
    Fire(STPLS3DPreprocessing)
