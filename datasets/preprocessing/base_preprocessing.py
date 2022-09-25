import os
import sys
import re
import yaml
import json
import multiprocessing
from pathlib import Path
from hashlib import md5

import numpy as np
from fire import Fire
from tqdm import tqdm
from joblib import Parallel, delayed
from loguru import logger


class BasePreprocessing:
    def __init__(
        self,
        data_dir: str = "./data/raw/",
        save_dir: str = "./data/processed/",
        modes: tuple = ("train", "validation", "test"),
        n_jobs: int = -1,
    ):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.n_jobs = n_jobs
        self.modes = modes

        if not self.data_dir.exists():
            logger.error("data folder doesn't exist")
            raise FileNotFoundError
        if self.save_dir.exists() is False:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.files = {}
        for data_type in self.modes:
            self.files.update({data_type: []})

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
                database.append(filebase)
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
                database.append(filebase)
            self.save_database(database, mode)
        self.fix_bugs_in_labels()
        self.joint_database()
        self.compute_color_mean_std(
            train_database_path=(self.save_dir / "train_database.yaml")
        )

    def process_file(self, filepath, mode):
        """process_file.

        Args:
            filepath: path to the main file
            mode: typically train, test or validation

        Returns:
            filebase: info about file
        """
        raise NotImplementedError

    def make_instance_database_sequential(
        self,
        train_database_path: str = "./data/processed/train_database.yaml",
        mode="instance",
    ):
        train_database = self._load_yaml(train_database_path)
        instance_database = []
        for sample in tqdm(train_database):
            instance_database.append(self.extract_instance_from_file(sample))
        self.save_database(instance_database, mode=mode)

    @logger.catch
    def make_instance_database(
        self,
        train_database_path: str = "./data/processed/train_database.yaml",
        mode="instance",
    ):
        self.n_jobs = multiprocessing.cpu_count() if self.n_jobs == -1 else self.n_jobs
        train_database = self._load_yaml(train_database_path)
        instance_database = []
        logger.info(f"Files in database: {len(train_database)}")
        parallel_results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self.extract_instance_from_file)(sample)
            for sample in train_database
        )
        for filebase in parallel_results:
            instance_database.append(filebase)
        self.save_database(instance_database, mode=mode)

    def extract_instance_from_file(self, sample_from_database):
        points = np.load(sample_from_database["filepath"])
        labels = points[:, -2:]
        file_instances = []
        for instance_id in np.unique(labels[:, 1]):
            occupied_indices = np.isin(labels[:, 1], instance_id)
            instance_points = points[occupied_indices].copy()
            instance_classes = np.unique(instance_points[:, 9]).astype(int).tolist()

            hash_string = str(sample_from_database["filepath"]) + str(instance_id)
            hash_string = md5(hash_string.encode("utf-8")).hexdigest()
            instance_filepath = self.save_dir / "instances" / f"{hash_string}.npy"
            instance = {
                "classes": instance_classes,
                "instance_filepath": str(instance_filepath),
                "instance_size": len(instance_points),
                "original_file": str(sample_from_database["filepath"]),
            }
            if not instance_filepath.parent.exists():
                instance_filepath.parent.mkdir(parents=True, exist_ok=True)
            np.save(instance_filepath, instance_points.astype(np.float32))
            file_instances.append(instance)
        return file_instances

    def fix_bugs_in_labels(self):
        pass

    def compute_color_mean_std(
        self, train_database_path: str = "./data/processed/train_database.yaml",
    ):
        pass

    def save_database(self, database, mode):
        for element in database:
            self._dict_to_yaml(element)
        self._save_yaml(self.save_dir / (mode + "_database.yaml"), database)

    def joint_database(self, train_modes=["train", "validation"]):
        joint_db = []
        for mode in train_modes:
            joint_db.extend(self._load_yaml(self.save_dir / (mode + "_database.yaml")))
        self._save_yaml(self.save_dir / "train_validation_database.yaml", joint_db)

    @classmethod
    def _read_json(cls, path):
        with open(path) as f:
            file = json.load(f)
        return file

    @classmethod
    def _save_yaml(cls, path, file):
        with open(path, "w") as f:
            yaml.safe_dump(file, f, default_style=None, default_flow_style=False)

    @classmethod
    def _dict_to_yaml(cls, dictionary):
        if not isinstance(dictionary, dict):
            return
        for k, v in dictionary.items():
            if isinstance(v, dict):
                cls._dict_to_yaml(v)
            if isinstance(v, np.ndarray):
                dictionary[k] = v.tolist()
            if isinstance(v, Path):
                dictionary[k] = str(v)

    @classmethod
    def _load_yaml(cls, filepath):
        with open(filepath) as f:
            file = yaml.safe_load(f)
        return file


if __name__ == "__main__":
    Fire(BasePreprocessing)
