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

        l.debug(
            "finished common parsing: {} classes, {} features".format(
                self.nclasses, self.nfeatures
            )
        )

    def _build_rule(self, data, fletter="X", delim="<="):
        """A list of lists of lower and upper bounds on all features.

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

            rule.append([minval, maxval])
            # some features preceding `ftridx` might have been skipped
            skip = [[minv, maxv] for i in range(len(rule) + 1, ftridx)]
            rule = skip + rule

        # some features following `ftridx` might not be present
        lacking = [
            [minv, maxv] for i in range(len(rule), self.nfeatures)
        ]
        rule = rule + lacking

        return rule



class RulesParser():

    def __init__(self, fname):
        pass


class ClassRulesParser(ReportParser):

    def __init__(self, fname):
        super().__init__(fname)
        l = logging.getLogger(__name__)
        l.debug("RulesClassParser.__init__()")

        cnamep = re.compile("^Класс [\d]*$")
        cnames = [s for s in self.soup.strings if re.match(cnamep, s)]
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


class RulesReport():

    def __init__(self):
        super().__init__()

        super().info("RulesReport __init__() started")

        pattern = "(\d*(\.\d*)?<=)?X\d*(<=\d*(\.\d*)?)?"
        self._rule_pattern = re.compile(pattern)

        super().info("RulesReport __init__() finished")

    def _parse_rule(self, data):
        if self._cur_rule >= self._num_rules:
            super().critical("There are too many rules")
            return

        if data.startswith("Закономерность"):
            label = re.search("\(класс \d*\)$", data)
            if label == None:
                super().critical("Could not find out class")
            else:
                label = label.group().strip("()")
                if self._cur_label != label:
                    # this is a new class label --- create a key in
                    # the rules dictionary for it
                    self._cur_label = label
                    self._rules[label] = []

                    super().info("Creating new label {}".format(label))
        else:
            rule = self._build_rule(data)
            self._rules[self._cur_label].append(rule)

            self._cur_rule += 1
            if self._cur_rule == self._num_rules:
                self._substate = "parse rules done"

                msg = [self._state, self._substate]
                super().info(
                    "Entering state {}, substate {}".format(*msg)
                )

    def _handle_rule_table(self, data):
        if (self._state == "rule table" and
            self._substate == "default" and
            data == "Количество закономерностей"):
            self._substate = "num of rules"

            msg = [self._state, self._substate]
            super().info("Entering state {}, substate {}".format(*msg))
        elif self._substate == "num of rules":
            self._num_rules = int(data)
            self._cur_rule = 0

            # create a special minmax rule which will hold the broadest
            # among the encountered ranges for each conjunction
            minfloat, maxfloat = self._minv, self._maxv
            self._rules["minmax_rule"] = [
                [maxfloat, minfloat] for i in range(self._num_features)
            ]

            self._substate = "parse rules"

            msg = [self._state, self._substate]
            super().info("Entering state {}, substate {}".format(*msg))
        elif self._substate == "parse rules":
            self._parse_rule(data)

    def handle_data(self, data):
        super().handle_data(data)
        if (self._state == "general info parsed" and
            data == "Найденные закономерности"):
            self._state = "rule table"

            msg = [self._state, self._substate]
            super().info("Entering state {}, substate {}".format(*msg))
        elif self._state == "rule table":
            self._handle_rule_table(data)


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
        pprint.pprint(ClassRulesParser(cmd.fname).rules)
    elif cmd.lrules:
        assert False
    else:
        assert False
