#!/usr/bin/env python3

import logging
import numpy as np

from sklearn.cluster import KMeans

from log import logsettings
from rulstat import RulesStats

class NRules:
    def __init__(self, i, n_clusters=2):
        self.km = KMeans(n_clusters=n_clusters)
        self.i, self.thresholds = i, np.linspace(0.1, 0.9, 36)

    def fit(self, X, y=None):
        self.km.fit(X, y)

    def restore(self, data, labels, criterion=RulesStats.infogain):
        """data is an np.array"""
        assert self.km is not None

        logger = logging.getLogger(__name__)

        self.cluster_centers_ = []
        for center in self.km.cluster_centers_:
            infovals = []
            for thresh in self.thresholds:
                mask = center > thresh
                if np.all(mask == 0):
                    infovals.append(0)
                    continue

                stats = {}
                for label in np.unique(labels):
                    same_label_mask = labels == label
                    stats[label] = [
                        np.sum(mask[same_label_mask] == 1),
                        np.sum(mask[same_label_mask] == 0),
                    ]

                infovals.append(
                    criterion(
                        [stats[k] for k in stats.keys()]
                    )
                )

            mask = center > self.thresholds[np.argmax(infovals)]
            # assert np.any(mask)
            if np.any(mask):

                minvals = np.min(data[mask, :], axis=0)
                maxvals = np.max(data[mask, :], axis=0)

                self.cluster_centers_.append(
                    np.hstack(
                        [(i, j) for i, j in zip(minvals, maxvals)]
                    )
                )
            else:
                logger.warning("cluster center is inadequate")

if __name__ == "__main__":
    import logging.config

    logging.config.dictConfig(logsettings)
