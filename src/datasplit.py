#!/usr/bin/env python3

import logging
import numpy as np

from sklearn.datasets.mldata import fetch_mldata
from sklearn.cross_validation import StratifiedKFold

### custom imports follow ###
from tabpar import TabDataParser

if __name__ == "__main__":
    import os.path
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument(
        "--data-home", nargs="?", default=os.path.join("../", "data"),
        help="path to folder containing mldata folder",
    )
    parser.add_argument(
        "--prefix", nargs="?", default=None,
        help="prepend to all resulring files",
    )
    parser.add_argument(
        "--target-name", nargs="?", default="label",
        help="name of the column containing the target values"
    )

    parsed = parser.parse_args()

    data_home = parsed.data_home
    if parsed.dataset == "climate-model-simulation-crashes":
        target_name = "int3"
        if parsed.target_name != target_name:
            logging.warning(
                "{} target is {}".format(parsed.dataset, target_name)
            )
        parsed.target_name = target_name

    bunch = fetch_mldata(
        parsed.dataset, target_name=parsed.target_name,
        data_home=data_home
    )

    data, labels = bunch['data'], bunch['target']
    old_labels = np.empty_like(labels)
    np.copyto(old_labels, labels)
    for i, label in enumerate(np.unique(labels)):
        labels[old_labels == label] = i + 1
    labels = np.ravel(labels).astype(int)

    skf = StratifiedKFold(
        y=labels, n_folds=2, shuffle=False, random_state=42
    )
    # get the last of the two splits
    for train_idx, test_idx in skf: pass

    if parsed.prefix is None:
        prefix = parsed.dataset
    else:
        prefix = parsed.prefix

    ftrain = os.path.join(data_home, "{}-train.tab".format(prefix))
    ftest = os.path.join(data_home, "{}-test.tab".format(prefix))
    fall = os.path.join(data_home, "{}-all.tab".format(prefix))

    TabDataParser.np2tab(ftrain, data[train_idx, :], labels[train_idx])
    TabDataParser.np2tab(ftest, data[test_idx, :], labels[test_idx])
    TabDataParser.np2tab(fall, data, labels)
