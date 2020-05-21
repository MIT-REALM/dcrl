from consMDP import ConsMDP

def product(cmdp, capacity, targets=[]):
    """Explicit encoding of energy into state-space

    The state-space of the newly created MDP consists of tuples $(s, e)$,
    where $s$ is the state of the input CMDP and $e$ is the energy level.
    For a tuple-state $(s,e)$ and an action $a$ with consumption (in the
    input CMDP) $c$, all successors of the action $a$ in the new MDP are
    of the form $(s', e-c)$ for non-reload states and
    $(r,\mathsf{cap})$ for reload states.
    """
    result = ConsMDP()

    # This will be our state dictionary
    sdict = {}
    # The list of output states for which we have not yet
    # computed the successors.  Items on this list are triplets
    # of the form `(s, e, p)` where `s` is the state
    # number in the mdp, `e` is the energy level, and p
    # is the state number in the output mdp.
    todo = []
    otargets = []
    sink_created = False

    # Transform a pair of state numbers (s, e) into a state
    # number in the output mdp, creating a new state if needed.
    # Whenever a new state is created, we can add it to todo.
    def dst(s, e):
        pair = (s, e)
        p = sdict.get(pair)
        if p is None:
            p = result.new_state(name=f"{s},{e}",
                                 reload=cmdp.is_reload(s))
            sdict[pair] = p
            if s in targets and e >= 0:
                otargets.append(p)
            todo.append((s, e, p))
        return p

    # Initialization
    # For each state of mdp add a new initial state
    for s in range(cmdp.num_states):
        dst(s, capacity)

    # Build all states and edges in the product
    while todo:
        s, e, p = todo.pop()
        for a in cmdp.actions_for_state(s):
            # negative goes to sink
            if e - a.cons < 0:
                if not sink_created:
                    sink = result.new_state(name="sink, -∞")
                    result.add_action(sink, {sink: 1}, "σ", 1)
                    sink_created = True
                result.add_action(p, {sink: 1}, a.label, a.cons)
                continue
            # build new distribution
            odist = {}
            for succ, prob in a.distr.items():
                new_e = capacity if cmdp.is_reload(succ) else e - a.cons
                out_succ = dst(succ, new_e)
                odist[out_succ] = prob
            result.add_action(p, odist, a.label, a.cons)

    return result, otargets