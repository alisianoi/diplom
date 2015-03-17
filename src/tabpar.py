#!/usr/bin/env python3

import numpy as np
import logging, logging.config

### custom imports follow ###
from log import logsettings


class TabDataParser:
    """Parser for Recognition-specific TAB data format."""

    def __init__(self, fname):

        log = logging.getLogger(__name__)

        self.fname = fname

        with open(fname) as src:
            header = [int(i) for i in src.readline().split()]

            self.nfeatures = header[0]
            self.nclasses  = header[1]
            cumcount       = header[2 : -1]
            self.NaN       = header[-1]

            log.debug(
                "{} classes, {} features".format(
                    self.nclasses, self.nfeatures
                )
            )

            self.data = dict()

            lines = (line for line in (l.strip() for l in src) if line)

            label = 1
            self.data[label] = []
            for i, line in enumerate(lines):
                x = tuple(float(feature) for feature in line.split())
                assert len(x) == self.nfeatures
                if self.NaN in x: log.warning("holes in data")
                self.data[label].append(x)
                if i + 1 == cumcount[label]:
                    l, ll = label, len(self.data[label])
                    log.debug("label {}, {} objects".format(l, ll))

                    if i + 1 != cumcount[-1]:
                        label += 1
                        self.data[label] = []

            for i in lines:
                log.warning("there is data left in file: {}".format(i))
                assert False

    def np2tab(self, fname, data, labels, nan="-1"):
        cls = np.unique(labels)
        [N, D] = data.shape

        with open(fname, 'w') as dst:
            l = " ".join(
                [str(i) for i in np.cumsum(np.binocount(labels))]
            )
            print("{} {}".format(D, N) + " " + l + " " + nan, file=dst)

            for x in data:
                print(" ".join([str(i) for i in x]), file=dst)



if __name__ == "__main__":
    import os, argparse

    description = "Tab Format Parser"
    aap = argparse.ArgumentParser(description=description)

    aap.add_argument("fname", help="name of *.tab file")

    cmd = aap.parse_args()

    logging.config.dictConfig(logsettings)

    tdp = TabDataParser(cmd.fname)
    print(tdp.data[1][-1])
