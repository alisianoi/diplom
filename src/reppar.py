#!/usr/bin/env python3

import re, sys
import logging, logging.config
import pprint

import numpy as np

from bs4 import BeautifulSoup

### custom imports follow ###
from log import logsettings


class ReportParser():

    def __init__(self, fname):
        l = logging.getLogger(__name__)

        with open(cmd.fname, "r", encoding="cp1251") as src:
            self.soup = BeautifulSoup(src)

        tr_features = self.soup.find(
            text=re.compile("^Пространство$")
        ).find_next("tr")
        tr_classes = tr_features.next_sibling

        self.nfeatures = int(list(tr_features.strings)[1])
        self.nclasses = int(list(tr_classes.strings)[1])
        assert self.nclasses >= 1 and self.nfeatures >= 1

        # this will be a dictionary of rules: class is key, value is a
        # list of rules for that class.
        self.rules = {}

        l.debug(
            "finished common parsing: {} classes, {} features".format(
                self.nclasses, self.nfeatures
            )
        )

    def _build_rule(self, data, fletter="X", delim="<="):
        """A tuple of tuples of lower and upper bounds on all features.

        Builds a full rule. Assumes that features are sorted.

        data is the raw string with the rule
        fletter is the letter to denote `feature`
        delim is supposed to be one of <, >, <=, >=
        """

        l = logging.getLogger(__name__)

        [minv, maxv] = [-np.Inf, +np.Inf]

        # digit-dot-digit
        ddd = "\d*(\.\d*)?"
        # space-delim-space
        sds = "\s*{}\s*".format(delim)
        rulep = re.compile(
            "({0}{1})?{2}\d*({1}{0})?".format(ddd, sds, fletter)
        )

        # lfl stands for `length of feature letter`
        rule, x, lfl = [], fletter, len(fletter)
        # feature_rng stands for feature_range --- the range in which
        # the value of that feature should be, according to this rule
        i = 0
        for feature_range in rulep.finditer(data):
            feature_range = feature_range.group().split(delim)
            feature_range = [f.strip() for f in feature_range]
            if len(feature_range) == 3: # both bounds present
                j = int(feature_range[1][lfl:])
                minval = float(feature_range[0][:])
                maxval = float(feature_range[2][:])
            elif feature_range[0].startswith(x): # upper bound
                j= int(feature_range[0][lfl:])
                minval, maxval = minv, float(feature_range[1][:])
            elif feature_range[1].startswith(x): # lower bound
                j = int(feature_range[1][lfl:])
                minval, maxval = float(feature_range[0][:]), maxv
            else:
                l.critical("Problem parsing feature range")
                sys.exit()

            # some features preceding `j` might have been skipped
            skip, i = [(minv, maxv) for j in range(i + 1, j)], j
            # l.debug("{} + {} + {}".format(len(rule), len(skip), 1))
            rule = rule + skip + [(minval, maxval)]

        # some features following `ftridx` might not be present
        lacking = [
            (minv, maxv) for k in range(i, self.nfeatures)
        ]
        # l.debug("{} + {} + {}".format(len(rule), len(lacking), 0))
        rule = rule + lacking

        assert len(rule) == self.nfeatures
        return tuple(rule)


class RulesParser(ReportParser):

    def __init__(self, fname):
        super().__init__(fname)
        l = logging.getLogger(__name__)
        l.debug("RulesClassParser.__init__()")

        rtables = self.soup.find_all(
            text=re.compile("^Найденные закономерности$")
        )
        assert len(rtables) >= 1

        for t in rtables:
            rules = self._table2rules(t)

            for key in rules.keys():
                if key in self.rules.keys():
                    l.debug("class {}, there were {} rules".format(
                        key, len(self.rules[key]))
                    )
                    for rule in rules[key]:
                        if rule not in self.rules[key]:
                            self.rules[key].append(rule)
                    l.debug("class {}, now there are {} rules".format(
                        key, len(self.rules[key]))
                    )
                else:
                    self.rules[key] = rules[key]

        for key in self.rules.keys():
            self.rules[key] = list(self.rules[key])

    def _table2rules(self, rtable):
        l = logging.getLogger(__name__)
        l.debug("RulesParser._table2rules()")

        tr_rule = rtable.find_next("tr")
        nrules = int([s for s in tr_rule.strings][1])
        l.info("there are a total of {} rules".format(nrules))

        rules, crules = {}, [] # `rules` shadows class instance
        rulep = re.compile("\(класс (\d)*\)$")

        tr_rule = tr_rule.next_sibling
        s = [s for s in tr_rule.strings]
        idx = int(rulep.search(s[0]).group(1))
        crules.append(self._build_rule(s[1]))

        for i in range(1, nrules):
            tr_rule = tr_rule.next_sibling

            s = [s for s in tr_rule.strings]
            nidx = int(rulep.search(s[0]).group(1))
            if (idx == nidx): # this rule has the same class as before
                crules.append(self._build_rule(s[1]))
            else:
                l.info(
                    "switching classes: {} to {} on rule {}".format(
                        idx, nidx, i
                    )
                )
                rules[idx] = crules
                crules, idx = [], nidx
                l.info(
                    "class {}, {} candidate rules".format(
                        idx - 1, len(rules[idx - 1])
                    )
                )

        # rules of the final class
        rules[idx] = crules
        l.debug(
            "class {}, {} candidate rules".format(
                idx, len(rules[idx])
            )
        )

        return rules


class ClassRulesParser(ReportParser):

    def __init__(self, fname):
        super().__init__(fname)
        l = logging.getLogger(__name__)
        l.debug("RulesClassParser.__init__()")

        cnames = self.soup.find_all(text=re.compile("^Класс [\d]*$"))
        assert len(cnames) == self.nclasses

        self.rules = {}
        for i, cname in enumerate(cnames):
            l.debug("class {} out of {}:".format(i + 1, len(cnames)))

            # advance to the first rule of current cname
            tr_rule = cname.find_next("tr")

            crules = []
            # tr_rule.string is None if it's a rule
            while not tr_rule.string:
                s = [s for s in tr_rule.strings]
                assert len(s) == 4

                r = self._build_rule(s[2], fletter = "x", delim="<")
                assert len(r) == self.nfeatures
                crules.append(r)

                tr_rule = tr_rule.next_sibling

            self.rules[i + 1] = crules
            l.debug("{} rules for class {}".format(
                len(crules), i + 1)
            )


if __name__ == "__main__":
    import os, argparse

    description = "Parser for Recognition reports"
    aap = argparse.ArgumentParser(description=description)

    aap.add_argument("fname", help="name of Recognition report file")
    aap.add_argument("-v", action="count", help="verbosity level")

    g = aap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--lrules", action="store_true", help="logical rules"
    )
    g.add_argument(
        "--lclass", action="store_true", help="logical class rules"
    )

    cmd = aap.parse_args()

    logging.config.dictConfig(logsettings)

    if cmd.lclass:
        crp = ClassRulesParser(cmd.fname)
    elif cmd.lrules:
        rp = RulesParser(cmd.fname)
    else:
        assert False