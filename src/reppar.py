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
        l.debug("{} classes, {} features".format(
            self.nfeatures, self.nclasses)
        )
        assert self.nclasses >= 1 and self.nfeatures >= 1

        # this will be a dictionary of rules: key is a class, value is
        # a list of rules for that class.
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
        ftridx = 0
        for feature_range in rulep.finditer(data):
            feature_range = feature_range.group().split(delim)
            feature_range = [f.strip() for f in feature_range]
            if len(feature_range) == 3: # both bounds present
                ftridx = int(feature_range[1][lfl:])
                minval = float(feature_range[0][:])
                maxval = float(feature_range[2][:])
            elif feature_range[0].startswith(x): # upper bound
                ftridx= int(feature_range[0][lfl:])
                minval, maxval = minv, float(feature_range[1][:])
            elif feature_range[1].startswith(x): # lower bound
                ftridx = int(feature_range[1][lfl:])
                minval, maxval = float(feature_range[0][:]), maxv
            else:
                l.critical("Problem parsing feature range")
                sys.exit()

            rule.append((minval, maxval))
            # some features preceding `ftridx` might have been skipped
            skip = [(minv, maxv) for i in range(len(rule) + 1, ftridx)]
            rule = skip + rule

        # some features following `ftridx` might not be present
        lacking = [
            (minv, maxv) for i in range(len(rule), self.nfeatures)
        ]
        rule = rule + lacking

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
                assert isinstance(rules[key], set)
                if key in self.rules.keys():
                    l.debug("there were {} rules".format(
                        len(self.rules[key]))
                    )
                    self.rules[key].union(rules[key])
                    l.debug("there are {} rules".format(
                        len(self.rules[key]))
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
        l.debug("there are a total of {} rules".format(nrules))

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
                l.debug(
                    "switching classes: {} to {} on rule {}".format(
                        idx, nidx, i
                    )
                )
                rules[idx] = set(tuple(crules))
                crules, idx = [], nidx
                l.debug(
                    "class {} adding {} rules".format(
                        idx - 1, len(rules[idx - 1])
                    )
                )

        # rules of the final class
        rules[idx] = set(tuple(crules))
        l.debug(
            "class {} adding {} rules".format(idx, len(rules[idx]))
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
            l.debug("{} out of {}".format(i + 1, len(cnames)))

            # advance to the first rule of current cname
            tr_rule = cname.find_next("tr")

            crules = []
            # tr_rule.string is None if it's a rule
            while not tr_rule.string:
                s = [s for s in tr_rule.strings]
                assert len(s) == 4

                crules.append(
                    self._build_rule(s[2], fletter = "x", delim="<")
                )

                tr_rule = tr_rule.next_sibling

            self.rules[i] = crules
            l.debug("{} rules for class {}".format(
                len(self.rules), i + 1)
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
        for key in crp.rules.keys(): print(type(crp.rules[key]))
        print(crp.rules[key][0])
    elif cmd.lrules:
        rp = RulesParser(cmd.fname)
        for key in rp.rules.keys(): print(type(rp.rules[key]))
        print(rp.rules[key][0])
    else:
        assert False
