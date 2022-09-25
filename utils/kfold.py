""" Author: https://github.com/yk-szk/stratified_group_kfold """
import random
import numpy as np


class StratifiedGroupKFold:
    """
    Stratified Group K-fold with sklearn.model_selection.KFold compabitility.

    Split dataset into k folds with balanced label distribution (stratified) and non-overlapping group.

    Args:
        n_splits (int): # of splits
        shuffle (bool): Shuffle
        seed (int): Seed value for random number generator
   """

    def __init__(self, n_splits, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = random_state

    def split(self, X, labels, groups):
        assert len(X) == len(labels) == len(groups), "Invalid input length"
        assert (
            len(set(groups)) >= self.n_splits
        ), "The number of groups needs to be larger than n_splits"

        def encode(v):
            s = set(v)
            d = {l: i for i, l in enumerate(s)}
            return [d[e] for e in v]

        labels, groups = encode(labels), encode(groups)
        num_labels, num_groups = max(labels) + 1, max(groups) + 1
        label_counts_per_group = np.zeros((num_groups, num_labels), dtype=int)
        global_label_dist = np.bincount(labels)
        for label, g in zip(labels, groups):
            label_counts_per_group[g][label] += 1

        label_counts_per_fold = np.zeros((self.n_splits, num_labels), dtype=int)
        groups_per_fold = [set() for _ in range(self.n_splits)]

        def eval_label_counts_per_fold(y_counts, fold):
            fold += y_counts
            std_per_label = np.std(label_counts_per_fold, axis=0) / global_label_dist
            fold -= y_counts
            return np.mean(std_per_label)

        groups_and_label_counts = list(enumerate(label_counts_per_group))
        if self.shuffle:
            rng = random.Random(self.seed)
            mean_std = np.mean(np.std(label_counts_per_group, axis=1))
            groups_and_label_counts.sort(
                key=lambda g_counts: -np.std(g_counts[1]) + rng.gauss(0, mean_std)
            )  # add rng.gauss to increase the randomness
        else:
            groups_and_label_counts.sort(key=lambda g_counts: -np.std(g_counts[1]))

        for g, label_counts in groups_and_label_counts:
            evals = [
                eval_label_counts_per_fold(label_counts, label_counts_per_fold[i])
                for i in range(self.n_splits)
            ]
            best_fold = np.argmin(evals)
            label_counts_per_fold[best_fold] += label_counts
            groups_per_fold[best_fold].add(g)

        all_groups = set(groups)
        for test_groups in groups_per_fold:
            train_groups = all_groups - test_groups

            train_indices = [i for i, g in enumerate(groups) if g in train_groups]
            test_indices = [i for i, g in enumerate(groups) if g in test_groups]

            yield train_indices, test_indices
