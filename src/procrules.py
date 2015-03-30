#!/usr/bin/env python3

import logging, copy

import numpy as np

### custom imports follow ###
from log import logsettings
from tabpar import TabDataParser
from reppar import RulesParser, ClassRulesParser
from misc import apply_rule


class ProcRules:

    def __init__(self, tabpar, reppar):
        logger = logging.getLogger(__name__)

        # data and rules are dicts; key is labels of class,
        # value is a list of lists of tuples of that class;
        self.data = copy.deepcopy(tabpar.data)
        self.rules = copy.deepcopy(reppar.rules)
        self.rulesbin = {}
        minv, maxv = reppar.minv, reppar.maxv

        for key in self.data.keys():
            assert key in self.rules.keys()
            assert not self.data[key] is None
            assert not self.rules[key] is None

        logger.debug("data and rules look fine")

        x = self.data[key][0]
        self.min, self.max = np.array(x), np.array(x)
        for y, X in self.data.items():
            for x in X:
                if tabpar.NaN in x: logger.warning("holes in data")

                xmin, xmax = np.array(x), np.array(x)

                self.min = np.min(np.array((self.min, xmin)), axis=0)
                self.max = np.max(np.array((self.max, xmax)), axis=0)

        self.minmax = np.append(
            [], [i for i in zip(self.min, self.max)]
        )

        logger.debug("minmax rule: {}".format(self.minmax))

        for key in self.rules.keys():
            l, f = len(self.rules[key]), len(self.rules[key][0])
            self.rules[key] = np.reshape(
                np.array(self.rules[key]), [l, 2 * f]
            )
            assert len(self.rules[key]) == l
            assert len(self.rules[key][0]) == 2 * f
            self.data[key] = np.array(self.data[key])

            for i in [-np.Inf, np.Inf]:
                m = self.rules[key] == i
                self.rules[key][m] = np.tile(self.minmax, (l, 1))[m]

        for rkey in self.rules.keys():
            self.rulesbin[rkey] = []
            for rule in self.rules[rkey]:
                binrule = []
                for xkey in self.data.keys():
                    for x in self.data[xkey]:
                        if apply_rule(rule, x):
                            binrule.append(1)
                        else:
                            binrule.append(0)

                self.rulesbin[rkey].append(binrule)

            self.rulesbin[rkey] = np.vstack(self.rulesbin[rkey])


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
