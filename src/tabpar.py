#!/usr/bin/env python3

import logging, logging.config

### custom imports follow ###
from log import logsettings


class TabDataParser:
    """Parser for Recognition-specific TAB data format."""

    def __init__(self, fname):

        log = logging.getLogger(__name__)

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

            for i in range(self.nclasses):
                self.data[i + 1] = []
                l, r = cumcount[i], cumcount[i + 1]
                for j in range(l, r):
                    x = tuple(float(k) for k in src.readline().split())
                    assert len(x) == self.nfeatures
                    if self.NaN in x: log.warning("holes in data")
                    self.data[i + 1].append(x)
                # each class finishes with a newline
                x = src.readline()
                log.debug(
                    "class {}, {} objects".format(
                        i + 1, len(self.data[i + 1])
                    )
                )


if __name__ == "__main__":
    import os, argparse

    description = "Tab Format Parser"
    aap = argparse.ArgumentParser(description=description)

    aap.add_argument("fname", help="name of *.tab file")

    cmd = aap.parse_args()

    logging.config.dictConfig(logsettings)

    tdp = TabDataParser(cmd.fname)
    print(tdp.data[1][-1])
