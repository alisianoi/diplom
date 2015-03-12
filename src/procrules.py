#!/usr/bin/env python3

import logging

import numpy as np

### custom imports follow ###
from log import logsettings
from tabpar import TabDataParser
from reppar import RulesParser, ClassRulesParser

class ProcRules():

    def __init__(self, tabpar, reppar):
        log = logging.getLogger(__name__)

        # data and rules are dicts; key is labels of class,
        # value is a list of lists of tuples of that class;
        self.data, self.rules = tabpar.data, reppar.rules
        minv, maxv = reppar.minv, reppar.maxv

        for key in self.data.keys():
            assert key in self.rules.keys()
            assert self.data[key] and self.rules[key]

        log.debug("data and rules look fine")

        x = self.data[key][0]
        self.min, self.max = np.array(x), np.array(x)
        for y, X in self.data.items():
            for x in X:
                if tabpar.NaN in x: log.warning("holes in data")

                xmin, xmax = np.array(x), np.array(x)

                self.min = np.min(np.array((self.min, xmin)), 0)
                self.max = np.max(np.array((self.max, xmax)), 0)

        self.minmax = np.append(
            [], [i for i in zip(self.min, self.max)]
        )

        log.debug("minmax rule: {}".format(self.minmax))

        for key in self.rules.keys():
            l, f = len(self.rules[key]), len(self.rules[key][0])
            self.rules[key] = np.reshape(
                np.array(self.rules[key]), [l, 2 * f]
            )
            self.data[key] = np.array(self.data[key])

            for i in [-np.Inf, np.Inf]:
                m = self.rules[key] == i
                self.rules[key][m] = np.tile(self.minmax, (l, 1))[m]


if __name__ == "__main__":
    import os, argparse, logging.config

    description = "Preprocess the rules"
    aap = argparse.ArgumentParser(description=description)

    aap.add_argument("tabfile", help="name of the *.tab data file")

    g = aap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--lrules", help="logical rules"
    )
    g.add_argument(
        "--lclass", help="logical class rules"
    )
    aap.add_argument("-v", action="count", help="verbosity level")

    cmd = aap.parse_args()

    logging.config.dictConfig(logsettings)

    tdp = TabDataParser(cmd.tabfile)

    if cmd.lclass:
        rp = ClassRulesParser(cmd.lclass)
    elif cmd.lrules:
        rp = RulesParser(cmd.lrules)
    else:
        assert False

    r = ProcRules(tdp, rp)
