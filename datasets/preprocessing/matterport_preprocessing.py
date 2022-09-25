import tempfile
import os
import re
import csv
import json
import multiprocessing
from pathlib import Path
import zipfile
from natsort import natsorted

from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
from fire import Fire
from loguru import logger

from mix3d.datasets.outdoor_semseg.base_preprocessing import BasePreprocessing
from mix3d.utils.point_cloud_utils import load_ply_with_normals


class MatterportPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir: str = "./data/raw/matterport/v1",
        save_dir: str = "./data/processed/matterport",
        modes: tuple = ("train", "validation", "test"),
        n_jobs: int = -1,
        git_repo: str = "./data/raw/matterport/Matterport",
        label_db: str = "configs/scannet_preprocessing/label_database.yaml",
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)

        git_repo = Path(git_repo)
        for mode in self.modes:
            trainval_split_dir = git_repo / "tasks" / "benchmark"
            matterport_special_mode = "val" if mode == "validation" else mode
            with open(
                trainval_split_dir / ("scenes_" + matterport_special_mode + ".txt")
            ) as f:
                # -1 because the last one is always empty
                split_file = f.read().split("\n")[:-1]

            filepaths = []
            for folder in split_file:
                filepaths.append(
                    self.data_dir / "scans" / folder / "region_segmentations.zip"
                )
            self.files[mode] = natsorted(filepaths)
        self.matterport_to_scannet_label = {}
        with open(git_repo / "metadata" / "category_mapping.tsv") as f:
            reader = csv.reader(f, delimiter="\t")
            columns = next(reader)
            raw_category = columns.index("raw_category")
            nyu40class = columns.index("nyu40class")
            for row in reader:
                self.matterport_to_scannet_label[row[raw_category]] = row[nyu40class]

        self.label_db = self._load_yaml(Path(label_db))
        # Originally it is "shower_curtain", but in matterport there is name only with space
        # other names that used for validation does not contain spaces
        self.label_db[28]["name"] = "shower curtain"

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
        with tempfile.TemporaryDirectory() as tempdir:
            with zipfile.ZipFile(filepath) as f:
                f.extractall(path=tempdir)

            region_files = (Path(tempdir) / scene_id).glob(r"*/*.ply")
            filebase = []
            for region_file in region_files:
                fbase = {
                    "filepath": "",
                    "raw_filepath": str(filepath),
                    "raw_filepath_in_archive": str(region_file),
                    "file_len": -1,
                }

                # reading both files and checking that they are fitting
                coords, features, _ = load_ply_with_normals(region_file)
                file_len = len(coords)
                fbase["file_len"] = file_len
                points = np.hstack((coords, features))

                if mode in ["train", "validation"]:
                    # getting instance info
                    instance_info_filepath = str(region_file).replace(
                        ".ply", ".semseg.json"
                    )
                    segment_indexes_filepath = str(region_file).replace(
                        ".ply", ".vsegs.json"
                    )
                    fbase["raw_instance_filepath"] = instance_info_filepath
                    fbase["raw_segmentation_filepath"] = segment_indexes_filepath
                    instance_db = self._read_json(instance_info_filepath)
                    segments = self._read_json(segment_indexes_filepath)
                    segments = np.array(segments["segIndices"])

                    # # adding instance label
                    labels = np.full((points.shape[0], 2), -1)
                    for instance in instance_db["segGroups"]:
                        segments_occupied = np.array(instance["segments"])
                        occupied_indices = np.isin(segments, segments_occupied)
                        labels[occupied_indices, 1] = instance["id"]
                        scannet_label = self.matterport_to_scannet_label.get(
                            instance["label"], -1
                        )
                        for k, v in self.label_db.items():
                            if v["name"] == scannet_label:
                                labels[occupied_indices, 0] = k
                                break
                    points = np.hstack((points, labels))

                region_num = int(re.search(r"\d+", region_file.stem).group(0))
                processed_filepath = (
                    self.save_dir / mode / f"{scene_id}_{region_num:02}.npy"
                )
                if not processed_filepath.parent.exists():
                    processed_filepath.parent.mkdir(parents=True, exist_ok=True)
                np.save(processed_filepath, points.astype(np.float32))
                fbase["filepath"] = str(processed_filepath)
                filebase.append(fbase)

        return filebase

    @logger.catch
    def preprocess(self):
        self.n_jobs = multiprocessing.cpu_count() if self.n_jobs == -1 else self.n_jobs
        for mode in self.modes:
            database = []
            logger.info(f"Tasks for {mode}: {len(self.files[mode])}")
            parallel_results = Parallel(n_jobs=self.n_jobs, verbose=10)(
                delayed(self.process_file)(file, mode) for file in self.files[mode]
            )
            for filebase in parallel_results:
                database.extend(filebase)
            self.save_database(database, mode)
        self.fix_bugs_in_labels()
        self.joint_database()
        self.compute_color_mean_std(
            train_database_path=(self.save_dir / "train_database.yaml")
        )

    def preprocess_sequential(self):
        for mode in self.modes:
            database = []
            for filepath in tqdm(self.files[mode], unit="file"):
                filebase = self.process_file(filepath, mode)
                database.extend(filebase)
            self.save_database(database, mode)
        self.fix_bugs_in_labels()
        self.joint_database()
        self.compute_color_mean_std(
            train_database_path=(self.save_dir / "train_database.yaml")
        )

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
    Fire(MatterportPreprocessing)
