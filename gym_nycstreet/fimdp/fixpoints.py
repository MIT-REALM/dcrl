"""
Core module for calculating the largest and the least fix point to be used
in energy_solver module for computing safe vector.
"""

from math import inf
from sys import stderr
debug = False
debug_vis = False


def argmin(iterable, func):
    """Compute argmin of func on iterable.

    Returns (i, v) such that v=func(i) is smallest in iterable.

    """

    res_item, res_val = None, inf
    for item in iterable:
        val = func(item)
        if val < res_val:
            res_item, res_val = item, val

    return res_item, res_val


def largest_fixpoint(mdp, values, action_value,
                     value_adj=lambda s, v: v,
                     skip_state=lambda x: False,
                     on_update=lambda s, v, a: None):
    """Largest fixpoint on list of values indexed by states.
    
    Most of the computations of energy levels are, in the end,
    using this function in one way or another.

    The value of a state `s` is a minimum over `action_value(a)`
    among all possible actions `a` of `s`. Values should be
    properly initialized (to ∞ or some other value) before calling.

    Parameters
    ==========
     * mdp      : `consMDP`
     * values   : `list of ints` values for the fixpoint

     * action_value : function that computes a value of an action
                      based on current values in `values`. Takes
                      2 paramers:
        - action    : `ActionData` action of MDP to evaluate
        - values    : `list of ints` current values

     * functions that alter the computation:
       - value_adj : `state × v -> v'` (default `labmda x, v: v`)
                      Change the value `v` for `s` to `v'` in each
                      iteration (based on the candidate value).
                      For example use for `v > capacity -> ∞`
                      Allows to handle various types of states
                      in a different way.

       - skip_state : `state -> Bool` (default `lambda x: False`)
                      If True, stave will be skipped and its value
                      not changed.

     * on_upadate : function called when new value for state is found.
                    Arguments are: state × value × action
                    The meaning is for `s` we found new value `v` using
                    action `a`.
                    By default only None is returned.

    Debug options
    =============
    We have 2 options that help us debug the code using this function:
     * `debug`     : print `values` at start of each iteration
     * `debug_vis` : display `mdp` using the IPython `display`
    """
    states = len(values)
    act_value = lambda a: action_value(a, values)

    # iterate until a fixpoint is reached
    iterate = True
    c = 0
    while iterate:
        if debug: print(f"it {c}\t:{values}", file=stderr)
        if debug_vis: display(f"Iteration {c}:", mdp.show("msrRb"))
        c += 1
        iterate = False

        for s in range(states):
            if skip_state(s):
                continue
            current_v = values[s]
            actions = mdp.actions_for_state(s)

            # candidate_v is the minimum over action values
            candidate_a, candidate_v = argmin(actions, act_value)

            # apply value_adj (capacity, reloads, ...)
            candidate_v = value_adj(s, candidate_v)

            # check for decrease in value
            if candidate_v < current_v:
                values[s] = candidate_v
                on_update(s, candidate_v, candidate_a)
                iterate = True


def least_fixpoint(mdp, values, action_value,
                     value_adj=lambda s, v: v,
                     skip_state=None):
    """Least fixpoint on list of values indexed by states.

    The value of a state `s` is a minimum over `action_value(a)`
    among all posible actions `a` of `s`. Values should be
    properly initialized (to ∞ or some other value) before calling.

    For safe values the values should be initialized to
    minInitCons.

    Parameters
    ==========
     * mdp      : `consMDP`
     * values   : `list of ints` values for the fixpoint

     * action_value : function that computes a value of an action
                      based on current values in `values`. Takes
                      2 paramers:
        - action    : `ActionData` action of MDP to evaluate
        - values    : `list of ints` current values

     * functions that alter the computation:
       - value_adj : `state × v -> v'` (default `labmda x, v: v`)
                      Change the value `v` for `s` to `v'` in each
                      iteration (based on the candidate value).
                      For example use for `v > capacity -> ∞`
                      Allows to handle various types of states
                      in a different way.

       - skip_state : `state -> Bool`
                      (default `lambda x: values[x] == inf`)
                      If True, stave will be skipped and its value
                      not changed.

    Debug options
    =============
    We have 2 options that help us debug the code using this function:
     * `debug`     : print `values` at start of each iteration
     * `debug_vis` : display `mdp` using the IPython `display`
    """
    if skip_state is None:
        skip_state = lambda x: values[x] == inf
    
    states = len(values)
    act_value = lambda a: action_value(a, values)

    # terate until a fixpoint is reached
    iterate = True
    c = 0
    while iterate:
        if debug: print(f"it {c}\t:{values}", file=stderr)
        if debug_vis: display(f"Iteration {c}:", mdp.show("msrRb"))
        c += 1
        iterate = False

        for s in range(states):
            if skip_state(s):
                continue

            current_v = values[s]
            actions = mdp.actions_for_state(s)
            # candidate_v is now the minimum over action values
            candidate_v = min([act_value(a) for a in actions])

            # apply value_adj (capacity, reloads, ...)
            candidate_v = value_adj(s, candidate_v)

            # least fixpoint increases only
            if candidate_v > current_v:
                values[s] = candidate_v
                iterate = True

