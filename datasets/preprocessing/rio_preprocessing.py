import os
import re
import csv
import json
from pathlib import Path
from hashlib import md5
from natsort import natsorted

import numpy as np
from fire import Fire
from tqdm import tqdm
from loguru import logger

from mix3d.datasets.preprocessing.base_preprocessing import BasePreprocessing
from mix3d.utils.point_cloud_utils import load_obj_with_normals


class RioPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir: str = "./data/raw/rio/rio",
        save_dir: str = "./data/processed/rio",
        modes: tuple = ("train", "validation", "test"),
        n_jobs: int = -1,
        git_repo: str = "./data/raw/rio/3RScan",
        label_db: str = "configs/scannet_preprocessing/label_database.yaml",
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)

        git_repo = Path(git_repo)
        self.files = {}
        for mode in self.modes:
            mode = "val" if mode == "validation" else mode
            trainval_split_dir = git_repo / "splits"
            with open(Path(trainval_split_dir) / (mode + ".txt")) as f:
                # -1 because the last one is always empty
                split_file = f.read().split("\n")[:-1]

            filepaths = []
            for folder in split_file:
                filepaths.append(self.data_dir / folder / "mesh.refined.obj")
            mode = "validation" if mode == "val" else mode
            self.files[mode] = natsorted(filepaths)

        self.rio_to_scannet_label = {}
        with open(git_repo / "data" / "mapping.tsv") as f:
            reader = csv.reader(f, delimiter="\t")
            columns = next(reader)
            raw_category = columns.index("Label")
            nyu40class = columns.index("NYU40 Mapping")
            for row in reader:
                self.rio_to_scannet_label[row[raw_category]] = row[nyu40class]

        self.label_db = self._load_yaml(Path(label_db))

    def process_file(self, filepath, mode):
        """process_file.

        Please note, that for obtaining segmentation labels json files were used.

        Args:
            filepath: path to the main ply file
            mode: train, test or validation

        Returns:
            filebase: info about file
        """
        scene_id = filepath.parent.name
        filebase = {
            "filepath": "",
            "raw_filepath": str(filepath),
            "file_len": -1,
        }

        # reading both files and checking that they are fitting
        coords, features = load_obj_with_normals(filepath)
        file_len = len(coords)
        filebase["file_len"] = file_len
        points = np.hstack((coords, features))

        if mode in ["train", "validation"]:
            # getting instance info
            instance_info_filepath = filepath.parent / "semseg.json"
            segment_indexes_filepath = next(filepath.parent.glob("*.segs.json"))
            instance_db = self._read_json(instance_info_filepath)
            segments = self._read_json(segment_indexes_filepath)
            segments = np.array(segments["segIndices"])
            filebase["raw_instance_filepath"] = instance_info_filepath
            filebase["raw_segmentation_filepath"] = segment_indexes_filepath

            # adding instance label
            labels = np.full((points.shape[0], 2), -1)
            for instance in instance_db["segGroups"]:
                segments_occupied = np.array(instance["segments"])
                occupied_indices = np.isin(segments, segments_occupied)
                labels[occupied_indices, 1] = instance["id"]
                scannet_label = self.rio_to_scannet_label.get(instance["label"], -1)
                for k, v in self.label_db.items():
                    if v["name"] == scannet_label:
                        labels[occupied_indices, 0] = k
                        break
            points = np.hstack((points, labels))

        processed_filepath = self.save_dir / mode / f"{scene_id}.npy"
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)
        return filebase

    def _read_json(self, path):
        try:
            with open(path) as f:
                file = json.load(f)
        except json.decoder.JSONDecodeError:
            with open(path) as f:
                # in some files I have wrong escapechars as "\o", while it should be "\\o"
                file = json.loads(f.read().replace(r"\o", r"\\o"))
        return file


if __name__ == "__main__":
    Fire(RioPreprocessing)
