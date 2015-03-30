def apply_rule(rule, x):
    assert len(rule) == 2 * len(x)

    t = 0
    for feature in x:
        if not (rule[t] <= feature and feature <= rule[t + 1]):
            return False
        t += 2

    return True
