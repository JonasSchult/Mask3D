import re
from pathlib import Path
from hashlib import md5
from natsort import natsorted

import numpy as np
from fire import Fire

from base_preprocessing import BasePreprocessing


class SemanticKittiPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir: str = "./data/raw/semantic_kitti",
        save_dir: str = "./data/processed/semantic_kitti",
        modes: tuple = ("train", "validation", "test"),
        n_jobs: int = -1,
        git_repo: str = "./data/raw/semantic-kitti-api",
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)

        git_repo = Path(git_repo)
        self.create_label_database(git_repo / "config" / "semantic-kitti.yaml")
        self.config = self._load_yaml(git_repo / "config" / "semantic-kitti.yaml")
        self.pose = dict()

        for mode in self.modes:
            scene_mode = "valid" if mode == "validation" else mode
            self.pose[mode] = dict()
            for scene in sorted(self.config["split"][scene_mode]):
                filepaths = list(self.data_dir.glob(f"*/{scene:02}/velodyne/*bin"))
                filepaths = [str(file) for file in filepaths]
                self.files[mode].extend(natsorted(filepaths))
                calibration = parse_calibration(
                    Path(filepaths[0]).parent.parent / "calib.txt"
                )
                self.pose[mode].update(
                    {
                        scene: parse_poses(
                            Path(filepaths[0]).parent.parent / "poses.txt", calibration,
                        ),
                    }
                )

    def create_label_database(self, config_file):
        if (self.save_dir / "label_database.yaml").exists():
            return self._load_yaml(self.save_dir / "label_database.yaml")
        config = self._load_yaml(config_file)
        label_database = {}
        for key, old_key in config["learning_map_inv"].items():
            label_database.update(
                {
                    key: {
                        "name": config["labels"][old_key],
                        # bgr -> rgb
                        "color": config["color_map"][old_key][::-1],
                        "validation": not config["learning_ignore"][key],
                    }
                }
            )

        self._save_yaml(self.save_dir / "label_database.yaml", label_database)
        return label_database

    def process_file(self, filepath, mode):
        """process_file.

        Args:
            filepath: path to the main ply file
            mode: train, test

        Returns:
            filebase: info about file
        """
        scene, sub_scene = re.search(r"(\d{2}).*(\d{6})", filepath).group(1, 2)
        filebase = {
            "filepath": filepath,
            "scene": int(scene),
            "sub_scene": int(sub_scene),
            "file_len": -1,
            "pose": self.pose[mode][int(scene)][int(sub_scene)].tolist(),
        }

        points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
        file_len = len(points)
        filebase["file_len"] = file_len

        if mode in ["train", "validation"]:
            # getting label info
            label_filepath = filepath.replace("velodyne", "labels").replace(
                "bin", "label"
            )
            filebase["label_filepath"] = label_filepath
            label = np.fromfile(label_filepath, dtype=np.uint32).astype(np.int32)
            if not points.shape[0] == label.shape[0]:
                raise ValueError("Files do not have same length")
            semantic_label = label & 0xFFFF
            instance_label = label >> 16

            semantic_label_copy = semantic_label.copy()
            for label in np.unique(semantic_label):
                semantic_label[semantic_label_copy == label] = self.config[
                    "learning_map"
                ][label]

            label = np.hstack(
                (semantic_label[:, np.newaxis], instance_label[:, np.newaxis])
            )
            points = np.hstack((points, label))

        processed_filepath = self.save_dir / mode / f"{scene}_{sub_scene}.npy"
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        return filebase


def parse_calibration(filename):
    """ read calibration file with given filename
        Returns
        -------
        dict
            Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}

    with open(filename) as calib_file:
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose
    return calib


def parse_poses(filename, calibration):
    """ read poses file with per-scan poses from given filename
        Returns
        -------
        list
            list of poses as 4x4 numpy arrays.
    """

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)

    with open(filename) as file:
        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

    return poses


if __name__ == "__main__":
    Fire(SemanticKittiPreprocessing)
