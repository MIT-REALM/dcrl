"""
Example related module used to generate Multi-agent grid world testbed that simulates
the motion of a rover and a helicopter operating on Martian surface with obstacles.
"""

import numpy as np
from numpy import linalg as LA
import consMDP
from decimal import Decimal

class MarsEnv:
    """Class that models a consumption Markov Decision Process with two agents following different dynamics. The 
    consumption MDP models the motion of a Martian rover and a aerial vehicle where the aerial vehicle can only reload
    when it occupies the same state as the rover. The aerial vehicle is referred to as agent.

    :param grid_size: Positive integer that denotes the size of the 2D grid to be generated. 
    :type grid_size: int
    :param agent_capacity: Non negative number denoting the energy capacity of the agent
    :type agent_capacity: float
    :param agent_actioncost: Positive number that denotes the energy consumed by the agent for any action other than stay, defaults to 1.
    :type agent_actioncost: float
    :param agent_staycost: Positive number that denotes the energy consumed by the agent for the stay action, defaults to 1.
    :type agent_staycost: float
    :param init_state: A tuple of length two denoting the initial state of the rover and the agent, defaults to None
    :type init_state: tuple, optional
    """

    def __init__(self, grid_size, agent_capacity, agent_actioncost=1, agent_staycost=1, init_state=None):
        """Constructor method
        """
        # User inputs and other required constants
        self.grid_size = grid_size
        self.agent_actioncost = agent_actioncost
        self.agent_staycost = agent_staycost
        self.agent_capacity = agent_capacity
        self.count = 0
        self.unreachable_cells = []
        if not isinstance(grid_size, int):
            raise TypeError("Grid size grid_size must be an integer greater than 0")

        # initialize required variables
        self.num_cells = grid_size**2
        self.num_actions = 5
        self.state = init_state
        self.agent_energy = self.agent_capacity

        # initialize grid and transition probabilities
        self.rover_P = np.zeros([self.num_cells, self.num_actions, self.num_cells])
        self.agent_P = np.zeros([self.num_cells, self.num_actions, self.num_cells])

        # generate dynamics for rover and agent
        # agent dynamics - deterministic. Rover dynamics - stochastic
        for i in range(self.num_cells):
            for j in range(self.num_actions):

                # north - action 0
                if j == 0:
                    if i - self.grid_size < 0:
                        self.rover_P[i,j,i] = 1
                        self.agent_P[i,j,i] = 1
                    else:
                        self.rover_P[i,j,i-self.grid_size] = 1
                        self.agent_P[i,j,i-self.grid_size] = 1

                # south - action 1
                elif j == 1:
                    if i + self.grid_size >= self.num_cells:
                        self.rover_P[i,j,i] = 1
                        self.agent_P[i,j,i] = 1
                    else:
                        self.rover_P[i,j,i+self.grid_size] = 1
                        self.agent_P[i,j,i+self.grid_size] = 1

                # east - action 2
                elif j == 2:
                    if i%self.grid_size == self.grid_size-1:
                        self.rover_P[i,j,i] = 1
                        self.agent_P[i,j,i] = 1
                    else:
                        self.rover_P[i,j,i+1] = 1
                        self.agent_P[i,j,i+1] = 1

                # west - action 3
                elif j == 3:
                    if i%self.grid_size == 0:
                        self.rover_P[i,j,i] = 1
                        self.agent_P[i,j,i] = 1
                    else:
                        self.rover_P[i,j,i-1] = 1
                        self.agent_P[i,j,i-1] = 1

                # stay - action 4
                elif j == 4:
                    self.rover_P[i,j,i] = 1
                    self.agent_P[i,j,i] = 1

                # rover stochastic dynamics sampled from dirichlet distribution
                if not ((i - self.grid_size < 0) or \
                        (i + self.grid_size >= self.num_cells) or \
                        (i%self.grid_size == self.grid_size-1) or \
                        (i%self.grid_size == 0)):

                    eps = 1e-10;
                    alpha = eps*np.ones(self.num_cells)
                    alpha.put([i, i-self.grid_size, i+self.grid_size,
                               i+1, i-1],2*np.ones(self.num_actions))
                    outcomes = [i-self.grid_size,i+self.grid_size,i+1,i-1,i]
                    alpha.put(outcomes[j],10)
                    if j == 4:
                        pass
                    else:
                        self.rover_P[i,j,:] = np.random.dirichlet(alpha, size=1)

        # generate unreachable cells
        self._unreachablestates()

        # initialize agent and rover positions
        self.reset(init_state)

    def _unreachablestates(self):
        """
        Internal method that generates a set of cells are made unreachable to the rover i.e. rover dynamics 
        are modified to make sure rover can never enter these cells irrespective of the action it takes.
        """

        if self.grid_size < 5:
            raise Exception("Grid size too small to generate unreachable cells; use a grid size no smaller than 5.")
        else:

            # declare unreachable cells
            centercell = [(self.grid_size-1)//2, (self.grid_size-1)//2]
            radius = (self.grid_size)//3

            for i in range(self.grid_size):
                boundary = []
                unreachable = []
                for i in range(self.num_cells):
                    # boundary cells
                    if LA.norm([i//self.grid_size-centercell[0], i%self.grid_size-centercell[1]]) <= radius:
                        boundary.append(i)
                    # unreachable cells
                    if LA.norm([i//self.grid_size-centercell[0], i%self.grid_size-centercell[1]]) <= radius-1:
                        unreachable.append(i)

            # change dynamics of boundary cells
            for cell in boundary:

                # new transiton array
                array = np.zeros(self.num_cells)
                array.put(cell,1)

                # north - 0
                if cell - self.grid_size in unreachable:
                    self.rover_P[cell, 0,:] = array
                    # Other actions
                    for i in [1,2,3]:
                        self.rover_P[cell, i, cell-self.grid_size] = 0
                    for i in [0,1,2,3]:
                        self.rover_P[cell, i,:] = self.rover_P[cell, i,:]/np.sum(self.rover_P[cell, i,:])

                # south - 1
                if cell + self.grid_size in unreachable:
                    self.rover_P[cell, 1,:] = array
                    # Other actions
                    for i in [0,2,3]:
                        self.rover_P[cell, i, cell+self.grid_size] = 0
                    for i in [0,1,2,3]:
                        self.rover_P[cell, i,:] = self.rover_P[cell, i,:]/np.sum(self.rover_P[cell, i,:])

                # east - 2
                if cell + 1 in unreachable:
                    self.rover_P[cell, 2,:] = array
                    # Other actions
                    for i in [0,1,3]:
                        self.rover_P[cell, i, cell+1] = 0
                    for i in [0,1,2,3]:
                        self.rover_P[cell, i,:] = self.rover_P[cell, i,:]/np.sum(self.rover_P[cell, i,:])

                # west - 3
                if cell - 1 in unreachable:
                    self.rover_P[cell, 3,:] = array
                    # Other actions
                    for i in [0,1,2]:
                        self.rover_P[cell, i, cell-1] = 0
                    for i in [0,1,2,3]:
                        self.rover_P[cell, i,:] = self.rover_P[cell, i,:]/np.sum(self.rover_P[cell, i,:])

            # change dynamics of unreachable cells
            for cell in unreachable:
                self.rover_P[cell,:,:] = np.zeros([self.num_actions, self.num_cells])
                self.rover_P[cell,:,cell] = np.ones(self.num_actions)

            # update unreachable cells
            self.unreachable_cells = unreachable

    def step(self, action):
        """
        Method that given action [agent action, rover action], function step updates the
        position of the rover and agent in the grid and also the energy of the
        agent. 
        
        Notes
        -----
        Stochastic actions inputs are also allowed.
        """

        # initialize variables
        done = 0
        previous_position = self.position

        # verify action input and decide action
        if np.shape(action) == (2,):
            pass
        elif np.shape(action) == (2,self.num_actions):
            if all(sum(act) == 1 for act in action):
                agent_act = np.random.choice(self.num_actions, p=action[0])
                rover_act = np.random.choice(self.num_actions, p=action[1])
                action = [agent_act, rover_act]
            else:
                raise Exception('Action probabilities do not add up to 1')
        else:
            raise Exception('Action dimensions are wrong. Please input action in one of the two suggested dimensions')

        ## update position
        
        # agent position
        self.position[0] = np.random.choice(self.num_cells,
                  p=self.agent_P[self.position[0],action[0],:])
        # rover position
        self.position[1] = np.random.choice(self.num_cells,
                  p=self.rover_P[self.position[1],action[1],:])

        ## update energy
        if previous_position[0] == self.position[0]:
            self.agent_energy -= self.agent_staycost
        else:
            self.agent_energy -= self.agent_actioncost
        # reload if feasible
        if self.position[0] == self.position[1]:
            self.agent_energy = self.agent_capacity
        ## out of energy flag
        if self.agent_energy == 0:
            done = 1
        elif self.agent_energy < 0:
            raise Exception('The energy of the agent cannot be negative.')

        # data export structure
        info = (tuple(self.position), tuple(action), self.agent_energy, done)
        return info

    def reset(self, init_state=None):
        """
        Method that resets the position of the rover and the agent and also the energy of the agent.
        It is ensured that the agent and the rover and reset to a feasible state. 
        """

        # random initial position
        if init_state == None:
            self.position = list(np.random.randint(self.num_cells,size=2))
        elif np.shape(init_state) == (2,):
            if all(0 <= state < self.num_cells for state in init_state) and \
            all(isinstance(state, int) for state in init_state):
                self.position = list(init_state)
            else:
                raise Exception('states should be integers between 0 and num_cells. Check init_state again')
        else:
            raise Exception('init_state should should be a 1X2 list')

        # override initial state if rover is in unreachable positions
        if self.position[1] in self.unreachable_cells:
            while True:
                self.position = list(np.random.randint(self.num_cells,size=2))
                if self.position[1] not in self.unreachable_cells:
                    break

        # reset agent energy
        self.agent_energy = self.agent_capacity

    def _get_targets(self, mdp):    
        """
        Internal method that generates target states that are well distributed over reachable
        and unreachable states and are a pre-defined proportion of the total no.of states. 
        """        
        
        target_prop = 0.2 # % of all states that will be targets
        
        # randomly select states
        T = set()
        state_list = mdp.names
        for state in state_list:
            if np.random.rand() <= target_prop:
                T.add(mdp.state_with_name(state))
        return T
        
        
    def _get_dist(self, mdp, state, action):
        """
        Internal method that returns a dictionary of states with nonzero probabilities for
        a given state and action
        """
        
        # Initialize
        dist = dict()
        agent_dist = self.agent_P[state[0], action[0], :]
        rover_dist = self.rover_P[state[1], action[1], :]
        agent_posstates = agent_dist.nonzero()
        rover_posstates = rover_dist.nonzero()
            
        # Store non-zero transition probabilities
        for i in list(agent_posstates[0]):
            for j in list(rover_posstates[0]):
                state = str((i,j))
                prob = agent_dist[i]*rover_dist[j]
                sid = mdp.state_with_name(str(state))
                dist[sid] = prob
    
        # Normalize to ensure probabilities add up exactly to 1
        s = sum(dist.values())
        for key in dist:
            dist[key] = round(Decimal(str(dist[key]/s)),4)
        key = list(dist.keys())[-1]
        dist[key] = Decimal(1) - sum(list(dist.values())[:-1])
        return dist
    
    def get_mdp_targets(self):
        """
        Method to export the martian gridworld and target states into a pre-defined
        standard form. Returns MDP object and the target set.
        """
        mdp = consMDP.ConsMDP()
    
        # Add states
        for i in range(self.num_cells):
            for j in range(self.num_cells):
                if i == j:
                    mdp.new_state(True, str((i, j)))  # (reload, label)
                else:
                    mdp.new_state(False, str((i, j)))
                
        # List all possible states and actions
        states_list = []
        for i in range(self.num_cells):
            for j in range(self.num_cells):
                states_list.append((i,j))
            
        actions_list = []
        for i in range(self.num_actions):
            for j in range(self.num_actions):
                actions_list.append((i,j))        

        # Extract and add actions to the MDP
        actions = dict()
        for state in states_list:
            for action in actions_list:
                action_label = str([state, action])
                dist = self._get_dist(mdp, state, action)
                if action[0] == action[1] == 4: # stay
                    actions[action_label] = {"from":str(state), "cons":self.agent_staycost, "dist":dist}
                else:
                    actions[action_label] = {"from":str(state), "cons":self.agent_actioncost, "dist":dist}
                
        for label, act in actions.items():
            fr = mdp.state_with_name(act["from"])
            mdp.add_action(fr, act["dist"], label, act["cons"])

        # Get targets
        target_set = self._get_targets(mdp)
    
        return (mdp, target_set)
