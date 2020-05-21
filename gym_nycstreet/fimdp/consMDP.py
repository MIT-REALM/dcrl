"""
Core module defining the class to model a consumption Markov Decision Process and
the associated methods. 
"""

from dot import consMDP2dot, dot_to_svg
from IPython.display import SVG
from energy_solver import EnergySolver, EnergyLevels_least
import math

def is_distribution(d):

    """Checks if a given array of probabilities is indeed a valid distribution

    Parameters
    ----------
    d: distribution; an array that sums up to 1.
       A valid probability distribution.

    Returns
    -------
    Boolean: Boolean Type
       0 or 1 depending on whether array d is a valid distribution.

    """
    probs = d.values()
    return sum(probs) == 1

class ConsMDP:
    """Represent Markov Decision Process with consumption on actions. The data describing 
       the MDP are stored mainly in two vectors:
         - `succ`
         - `actions`
    States are represented by integers and `succ[i]` stores the index to
    the list `actions` where the data for `i` start. Thus `actions[succ[i]]`
    hold the first action of `i`. To iterate over actions of a state `s`
    use `actions_for_state(s)`. If you wish to remove actions, use
    `out_iteraser(s)` instead.

    States can have names using the list `names`. Reload states are stored
    in the set `reload_states`.
    
    Computation of Safe vector
    ==========================
    The :math:`safe^{cap}` vector can be computed in 2 different ways.
    
    The variant used by default in consMDP can be controlled by def_EL_class.
    Currently, the default is 
    ```
    self.def_EL_class = EnergySolver
    ```
    The other option is `EnergyLevels_least`.
    
    The running times between the 2 variants can vary a lot, it hugely
    depends on the MDP and its structure. See notebook 
    [Safe-variants](Safe-variants.ipynb) for more details and comparison.
    
    Basically, EnergyLevels_least is faster on models where the maximal
    consumption on an action is strictly smaller than the number of states,
    and the other way.

    Important
    =========
    Functions that change the structure of the consMDP should always call
    `self.structure_change()`.

    Define your probabilities in distributions in some exact representation
    like `decimal.Decimal(probability_string)` and always avoid floating-point
    data types. Due to their imprecission some checks could fail or trigger
    false positives (e.g. `0.06+0.82+0.12 != 1`!).
    """

    def __init__(self):
        """Constructor method - consMDP class
        """
        self.name = None

        self.succ = []
        self.actions = [0]

        self.names = []
        self.names_dict = dict()
        self.reloads = []

        self.num_states = 0

        self.energy_levels = None
        self.def_EL_class = EnergySolver

    def structure_change(self):
        """Reset `energy_levels` to None after structure change."""
        self.energy_levels = None

    def state_with_name(self, name):
        '''Return id of state with name `name` or `None` if not exists.'''
        return self.names_dict.get(name)

    def new_state(self, reload=False, name=None):
        """Add a new state into the CMDP.

        Returns: the id of the created state
        Raise `ValueError` if a state with the same name already exists.
        """

        self.structure_change()

        # check for existing name
        if name is not None:
            s = self.state_with_name(name)
            if s is not None:
                raise ValueError("State with name \"{}\" already exists (id={})".
                                 format(name, s))
        
        sid = self.num_states
        
        self.succ.append(0)
        self.reloads.append(reload)
        self.names.append(name)
        if name is not None:
            self.names_dict[name] = sid
        self.num_states+=1
        return sid
    
    def new_states(self, count, names=None):
        #TODO add reload list
        """Create multiple (`count`) states.

        The list names must have length `count` if supplied. These will be
        the names for the states.

        Return the list of states ids.
        Raise `ValueError` if a state with the same name already exists.
        """
        if names is not None:
            if count != len(names):
                raise ValueError("Length of names must be equal to count.")

        self.structure_change()
        start = self.num_states
        for i in range(count):
            name = None if names is None else names[i]
            self.new_state(name)
        return range(start, start+count)

    def set_reload(self, sid, reload=True):
        #TODO extend to lists of states
        """Set reload status of state `sid`.

        Set to True by default."""
        self.structure_change()
        self.reloads[sid] = reload

    def unset_reload(self, sid):
        """Set the state `sid` *not* to be a reload state.

        Equivalent to
        ``
        set(sid, False)
        ``
        """
        self.structure_change()
        self.reloads[sid] = False

    def is_reload(self, sid):
        """Return the reload status of state `sid`."""
        return self.reloads[sid]

    def add_action(self, src, distribution, label, consumption = 0):
        """Add action to consMDP.

        Returns: index of the new action in the `actions` list.
        Raises: 
          * `ValueError` if attempt to use non-existent state
          * `ValueError` if src-label->... exists already. 
        """
        # Check that src exists
        if src >= self.num_states:
            raise ValueError(f"State {src} given as src does not exists.")

        # Check that all destinations exist
        for k in distribution.keys():
            if k >= self.num_states:
                raise ValueError(f"State {k} does not exists.")

        # check for determinism on action labels
        # raise ValueError if nondeterminsm would occur
        for a in self.actions_for_state(src):
            if a.label == label:
                raise ValueError(
                    "State {} already has an action with label {}".format(src, label))

        aid = len(self.actions)
        adata = ActionData(src, consumption, distribution, label, 0)

        self.structure_change()

        # Update the lists accordingly:
        #  * `next_succ` of last action for src if any, or
        #  * `succ[src]`  if it's first action for src
        a = self.actions_for_state(src).get_last()
        if a is None:
            self.succ[src] = aid
        else:
            a.next_succ = aid

        self.actions.append(adata)
        return aid

    def remove_action(self, aid):
        """Remove action based on its id."""
        # TODO add checks
        if aid <= 0 or aid >= len(self.actions):
            raise ValueError(f"The supplied aid {aid} is not a valid action id.")
        if self.actions[aid].next_succ == aid:
            raise ValueError(f"Action aid ({aid}) was already deleted.")

        self.structure_change()

        src = self.actions[aid].src
        it = self.out_iteraser(src)
        next(it)
        while it.curr != aid:
            next(it)
        it.erase()

    def actions_for_state(self, s):
        """Return iterator of actions available for state `s`."""
        it = Succ_iter(self.actions, self.succ[s])
        return it

    def out_iteraser(self, s):
        """Return iterator of actions available for state `s`."""
        it = Succ_iteraser(self, s)
        return it

    def state_succs(self, s):
        """Return successors of `s` over all actions"""
        succs = set()
        for e in self.actions_for_state(s):
            succs = succs.union(e.distr.keys())
        return succs

    def get_minInitCons(self,
                        capacity=None,
                        recompute=False,
                        compute_strategy=True):
        """Return (and store) the energy levels needed to reach some
        target within > 0 steps.
        
        If capacity is exceeded for state `s`, the value for `s` is ∞.

        By default use last capacity or ∞.
        """
        el = self.energy_levels
        if capacity is None:
            capacity = math.inf if el is None else el.cap
        if el is None or capacity != el.cap:
            recompute = True
        if recompute:
            self.energy_levels = self.def_EL_class(self, capacity)
        return self.energy_levels.get_minInitCons()

    def get_safe(self, capacity=None, recompute=False):
        """Return (and store) the energy levels needed to survive
        with given capacity.

        If cannot survival from `s` cannot be guaranteed with given
        capacity, the value for `s` is ∞.

        By default use last capacity or ∞.
        """
        self.get_minInitCons(capacity, recompute)
        return self.energy_levels.get_safe()

    def get_positiveReachability(self, targets,
                                 capacity=None, recompute=False):
        """Return (and store) the energy levels needed to reach T (`targets`)
        from each state.

        `targets` : set of ints
        `capacity`: capacity

        By default use last capacity or ∞.
        """
        el = self.energy_levels
        if capacity is None:
            capacity = math.inf if el is None else el.cap
        if el is None or el.cap != capacity or el.targets is None:
            recompute = True
        if recompute:
            self.energy_levels = self.def_EL_class(self, capacity, targets)
        return self.energy_levels.get_positiveReachability()

    def get_almostSureReachability(self, targets,
                                   capacity=None, recompute=False):
        """Return (and store) the energy levels needed to reach T (`targets`)
        from each state.

        `targets` : set of ints

        By default use last capacity or ∞.
        """
        el = self.energy_levels
        if capacity is None:
            capacity = math.inf if el is None else el.cap
        if el is None or el.cap != capacity or el.targets is None:
            recompute = True
        if recompute:
            self.energy_levels = self.def_EL_class(self, capacity, targets)
        return self.energy_levels.get_almostSureReachability()

    def get_Buchi(self, targets, capacity=None, recompute=False):
        """Return (and store) the energy levels needed to reach T (`targets`)
        infinitely often from each state.

        `targets` : set of ints

        By default use last capacity or ∞.
        """
        el = self.energy_levels
        if capacity is None:
            capacity = math.inf if el is None else el.cap
        if el is None or el.cap != capacity or el.targets is None:
            recompute = True
        if recompute:
            self.energy_levels = self.def_EL_class(self, capacity, targets)
        return self.energy_levels.get_Buchi()

    def get_dot(self, options=""):
        dwriter = consMDP2dot(self, options)
        return dwriter.get_dot()

    def show(self, options=""):
        return SVG(dot_to_svg(self.get_dot(options)))
        
    def _repr_dot_(self):
        return self.get_dot()

    def _repr_svg_(self):
        return dot_to_svg(self._repr_dot_())
        
        
class ActionData:
    """Object that holds the data of an action in the consMDP for a given
    state, consumption, distribution and label. 
    """
       
    def __init__(self, src, cons, distr, label, next_succ):
        if not is_distribution(distr):
            raise AttributeError("Supplied dict is not a distribution." +
                                 " The probabilities are: {}, sum: {}".format
                                 (list (distr.values()),
                                  sum(distr.values())) )
        self.src = src
        self.cons = cons
        self.distr = distr
        self.label = label
        self.next_succ = next_succ
        
    def get_succs(self):
        return set(self.distr.keys())
    

class Succ_iter:
    """Iterate over linked list nested in a given List.
    
    Expect that the items stored in `l` contain field `next_succ` which
    contain integers that are indices in `l` where new successor can be
    found.
    
    Assumptions
    ===========
     * List `l` is indexed from 1
     * `next_succ = 0` means last item of the linked list (`succ`)
     
    ..attribute: l
    
        List of items containing `next_succ` fields. Indexed from 1
        
    ..attribute: i
        
        Index in `l` of the first element of the nested list.
    
    """
    def __init__(self, l, i):
        """Constructor method - Succ_iter class
        """
        # TODO check for types
        self.l = l
        self.next = i

    def __iter__(self):
        return self

    def __next__(self):
        if self.next == 0:
            raise StopIteration()
        if self.next >= len(self.l):
            raise IndexError("{} ".format(self.l)+f"{self.next}")
        else:
            item = self.l[self.next]
            self.next = item.next_succ
            return item
        
    def is_empty(self):
        return self.next == 0
    
    def get_last(self):
        a = None
        while not self.is_empty():
            a = self.__next__()
        return a  
    
    def __len__(self):
        c = 0
        while not self.is_empty():
            a = self.__next__()
            c += 1
        return c

class Succ_iteraser(Succ_iter):
    """Iterate over outgoing edges of `s` and allow erasing edges.

    Expects an consMDP `mdp` and state index `s`. The function erase
    erases the current action and moves to the next one for `s`.
    """

    def __init__(self, mdp, s):
        """Constructor method - Succ_iteraser class
        """
        super(Succ_iteraser,self).__init__(mdp.actions, mdp.succ[s])
        self.curr = None
        self.prev = None
        self.s    = s
        self.succ = mdp.succ
        self.mdp = mdp

    def __next__(self):
        self.prev = self.curr
        self.curr = self.next
        return super(Succ_iteraser,self).__next__()

    def erase(self):
        if self.curr is None:
            raise ValueError("Can't erase before moved to 1st edge. Call self.__next__() first")

        self.mdp.structure_change()

        self.l[self.curr].next_succ = self.curr
        if self.prev is None:
            self.succ[self.s] = self.next
        else:
            self.l[self.prev].next_succ = self.next
