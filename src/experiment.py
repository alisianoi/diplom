#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.datasets.mldata import fetch_mldata
from sklearn.cross_validation import StratifiedKFold
from sklearn.cluster import KMeans

from tabpar import TabDataParser
from reppar import RulesParser
from procrules import ProcRules

from rulstat import RulesStats
from rcluster import NRules
from logical import SimpleVoting

### successful datasets:
# iris, wine, climate-model-simulation-crashes
# ionosphere as decent overfit example

if __name__ == "__main__":
    import logging, logging.config, os.path
    from argparse import ArgumentParser

    from log import logsettings
    logging.config.dictConfig(logsettings)
    logger = logging.getLogger(__name__)


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
    elif parsed.dataset == "uci-20070111-liver-disorders":
        target_name = "int2"
        if parsed.target_name != target_name:
            logging.warning(
                "{} target is {}".format(parsed.dataset, target_name)
            )
        parsed.target_name = target_name

    bunch = fetch_mldata(
        parsed.dataset, target_name=parsed.target_name,
        data_home=data_home
    )

    data, labels = scale(bunch['data']), bunch['target']
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

    frules = os.path.join(data_home, "{}-lrules.html".format(prefix))

    ftrain = os.path.join(
        data_home, "{}-train.tab".format(parsed.dataset)
    )
    ftest = os.path.join(
        data_home, "{}-test.tab".format(parsed.dataset)
    )
    fall = os.path.join(
        data_home, "{}-all.tab".format(parsed.dataset)
    )

    tab_parser = TabDataParser(ftrain)
    rules_parser = RulesParser(frules)

    logger.debug("about to process rules")
    processor = ProcRules(tab_parser, rules_parser)
    rules, rulesbin = processor.rules, processor.rulesbin
    logger.debug("rules processing finished")

    vote_mdl = SimpleVoting(rules)

    y = vote_mdl.fit(data[test_idx, :])

    full_correct = sum(
        [1 for i, j in zip(y, labels[test_idx]) if i == j]
    ) / len(labels[test_idx])

    logger.debug("full_correct: {}".format(full_correct))
    data_train = TabDataParser(ftrain)

    n_clusters = min([len(rules[k]) for k in rules.keys()])

    correct = []
    for i in range(2, n_clusters + 1):
        km = KMeans(n_clusters=i)
        nrules = {}
        for k in rules.keys():
            km.fit(rules[k])
            nrules[k] = km.cluster_centers_

        nvotemdl = SimpleVoting(nrules)
        y = nvotemdl.fit(data[test_idx, :])
        correct.append(
            sum(
                [1 for i, j in zip(y, labels[test_idx]) if i == j]
            ) / len(labels[test_idx])
        )

    igbincorrect = []
    for i in range(2, n_clusters + 1):
        nrules = {}
        for k in rulesbin.keys():
            km = NRules(i=k, n_clusters=i)
            km.fit(rulesbin[k])
            km.restore(
                data[train_idx, :], labels[train_idx],
                RulesStats.infogain
            )
            nrules[k] = km.cluster_centers_

        binvotemdl = SimpleVoting(nrules)
        y = binvotemdl.fit(data[test_idx, :])
        igbincorrect.append(
            sum(
                [1 for i, j in zip(y, labels[test_idx]) if i == j]
            ) / len(labels[test_idx])
        )

    stbincorrect = []
    for i in range(2, n_clusters + 1):
        nrules = {}
        for k in rulesbin.keys():
            km = NRules(i=k, n_clusters=i)
            km.fit(rulesbin[k])
            km.restore(
                data[train_idx, :], labels[train_idx],
                RulesStats.statcriterion
            )
            nrules[k] = km.cluster_centers_

        binvotemdl = SimpleVoting(nrules)
        y = binvotemdl.fit(data[test_idx, :])
        stbincorrect.append(
            sum(
                [1 for i, j in zip(y, labels[test_idx]) if i == j]
            ) / len(labels[test_idx])
        )

    plt.rcdefaults()
    plt.rc('text', usetex=True)
    plt.rc('text.latex', unicode=True)
    plt.rc('text.latex', preamble=r"\usepackage[utf8]{inputenc}")
    plt.rc('text.latex', preamble=r"\usepackage[russian]{babel}")
    plt.rcParams['font.serif'] = 'cmunst'

    x = list(range(2, n_clusters + 1))
    cutoff = [full_correct for i in range(2, n_clusters + 1)]
    plt.plot(
        x, cutoff, '-r', linewidth=2, label='простое голосование'
    )
    markersize=4
    plt.plot(
        x, correct, '-ob', label='вектор левых и правых границ',
        markersize=markersize
    )
    plt.plot(
        x, igbincorrect, '-^g', label='бинарный вектор, IGain',
        markersize=markersize
    )
    plt.plot(
        x, stbincorrect, '-oc', label='бинарный вектор, Stat',
        markersize=markersize
    )
    plt.legend(loc=4)
    plt.xlabel("количество логических закономерностей")
    plt.ylabel("доля верно классифицированных объектов")
    plt.savefig(
        "../LaTeX/graphs/{}.pdf".format(prefix), bbox_inches="tight"
    )
    plt.show()
