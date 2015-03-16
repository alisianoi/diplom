#!/usr/bin/env python3

import logging

import numpy as np

### custom imports follow ###
from log import logsettings
from tabpar import TabDataParser
from reppar import RulesParser, ClassRulesParser
from rulstat import apply_rule
from procrules import ProcRules


class SimpleVoting():
    def __init__(self, rules):
        self.rules = rules

    def fit(self, data):
        self.labels = []
        keys = self.rules.keys()
        for x in data:
            votes = {key : 0 for key in keys}
            for key in keys:
                for rule in self.rules[key]:
                    if apply_rule(rule, x):
                        votes[key] += 1
            votes = {
                key : votes[key] / len(self.rules[key]) for key in keys
            }

            # get the key with the highest raction of votes in favour
            self.labels.append(max(votes, key=votes.get))

        return self.labels

if __name__ == "__main__":
    pass
