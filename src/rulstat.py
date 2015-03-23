#!/usr/bin/env python3

import logging
import numpy as np

from math import log2
from scipy.special import binom

### custom imports follow ###
from log import logsettings
from procrules import ProcRules
from tabpar import TabDataParser
from reppar import RulesParser, ClassRulesParser


def apply_rule(rule, x):
    assert len(rule) == 2 * len(x)

    t = 0
    for feature in x:
        if not (rule[t] <= feature and feature <= rule[t + 1]):
            return False
        t += 2

    return True


class RulesStats():

    def __init__(self, rules):
        self.rules = rules
        self.stats = {}

    def compute_stats(self, data):
        logger = logging.getLogger(__name__)

        rules = self.rules
        self.stats = {}
        for rlabel in rules.keys():
            self.stats[rlabel] = []
            for rule in rules[rlabel]:
                stats = {}
                for xlabel in data.keys():
                    # number of accepted / rejected x by this rule
                    stats[xlabel] = [0, 0]
                    for x in data[xlabel]:
                        if apply_rule(rule, x):
                            stats[xlabel][0] += 1
                        else:
                            stats[xlabel][1] += 1

                # compute information gain; vokov, page 7
                I = RulesStats.statcriterion(
                    [stats[k] for k in stats.keys()]
                )
                if "I" in stats.keys():
                    logger.warning("statcritetion overwrite")
                stats["I"] = I

                self.stats[rlabel].append(stats)

    @staticmethod
    def statcriterion(contingency_table):
        x = [binom(sum(col), col[0]) for col in contingency_table]
        y = binom(
            sum([sum(col) for col in contingency_table]),
            sum([col[0] for col in contingency_table])
        )
        return np.prod(x) / y


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
    stats = RulesStats(r)
