"""
Example related module with benchmarking and visualization tools for AEV in NYC case study.
"""

import os
import ast
import folium
from folium import plugins
import networkx as nx
import timeit
import json
import numpy as np
import energy_solver
from energy_solver import BUCHI
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def timeit_difftargets(m, cap, target_size = 100, num_samples = 100, num_tests=5, obj=BUCHI):

    """Returns a list of compute times for given objective for the NYC AEV
    problem with varying target sets.

    Parameters
    ----------
    m : mdp;  object of class ConsMDP.
       A valid Markov Decision Process.

    cap : positive integer
       The energy capacity of the agent.

    target_size : positive integer, optional
       Size of the set of random targets to be generated for calculating the
       compute time variation. Takes a default value of 100.

    num_samples : positive integer, optional
       Number of times a random target set is generated and it's compute time
       for any given objective is calculated. Takes a default of 100.

    num_tests : positive integer, optional
       Number of times the compute time is calculated using timeit for each
       individual test. Takes a default value of 5.

    obj : one of MIN_INIT_CONS, SAFE, POS_REACH, AS_REACH, BUCHI, optional
       Objective for which the compute times are calculated. Takes BUCHI as
       the default value.

    Returns
    -------
    comptime : array_like
       List of computational times for different target sets.

    Examples
    --------
    >>> m, targets = ch_parser.parse('NYCstreetnetwork.json')
    >>> comptime = timeit_difftargets(m, cap=200, target_size=100)

    Notes
    -----
    The values of timings returned might significantly vary depending on the
    machine configuration and the target states set.

    """

    comptime = np.empty(num_samples)
    for i in range(num_samples):
        targets = np.random.randint(0, high=m.num_states, size=(target_size))
        def calc_time():
            s = energy_solver.EnergySolver(m, cap=cap, targets=targets)
            s.get_strategy(obj, recompute=True)
        comptime[i] = timeit.timeit(calc_time, number=num_tests)/num_tests
    return comptime


def timeit_diffcaps(m, targets, cap_bound , num_samples = 20, num_tests=10, obj=BUCHI):

    """Returns a list of tuples where each tuple consists of the energy capacity
    and its corresponding computational time for a given objective and target
    set in the NYC AEV problem.

    Parameters
    ----------
    m : mdp;  object of class ConsMDP
       A valid Markov Decision Process.

    targets : array_like, list of target states
       A list containing all the target states.

    cap_bound : positive integer
       Upper bound of the interval specifying the allowed energy capacity
       for the agent.

    num_samples : positive integer, optional
       Number of equally spaced samples to be collected from the interval
       [0, cap_bound] for evaluation of their computational time. Takes a
       default value of 20.

    num_tests : positive integer, optional
       Number of times the compute time is calculated using timeit for each
       sampled capacity value. Takes a default value of 10.

    obj : one of MIN_INIT_CONS, SAFE, POS_REACH, AS_REACH, BUCHI, optional
       Objective for which the compute times are calculated. Takes BUCHI as
       the default value.

    Returns
    -------
    comptime : array_like
       List with tuples of capacity and expected computational time for a fixed
       set of targets.

    Examples
    --------
    >>> m, targets = ch_parser.parse('NYCstreetnetwork.json')
    >>> comptime = timeit_diffcaps(m, targets=targets, cap_bound=200, obj=BUCHI)

    Notes
    -----
    The values of timings returned might significantly vary depending on the
    machine configuration.

    """

    comptime = []

    cap_list = np.linspace(1,cap_bound, num_samples,dtype = int)
    for i in range(num_samples):
        def calc_time():
            s = energy_solver.EnergySolver(m, cap=cap_list[i], targets=targets)
            s.get_strategy(obj, recompute=True)
        comptime.append((cap_list[i], timeit.timeit(calc_time, number=num_tests)/num_tests))
    return comptime

def timeit_difftargetsizes(m, cap, size_bound , num_samples = 20, num_tests=10, obj=BUCHI):

    """Returns a list of tuples where each tuple consists of the target size
        and its corresponding computational time for a given objective and capacity
        in the NYC AEV problem.

    Parameters
    ----------
    m : mdp;  object of class ConsMDP
       A valid Markov Decision Process.

    cap : positive integer.
       The energy capacity of the agent.

    size_bound : positive integer
       Upper bound of the interval specifying the allowed size of randomly
       generated target set for the agent.

    num_samples : positive integer, optional
       Number of equally spaced samples to be collected from the interval
       [20, size_bound] for evaluation of their computational time. Takes a
       default value of 20.

    num_tests : positive integer, optional
       Number of times the compute time is calculated using timeit for each
       target set. Takes a default value of 10.

    obj : one of MIN_INIT_CONS, SAFE, POS_REACH, AS_REACH, BUCHI, optional
       Objective for which the compute times are calculated. Takes BUCHI as
       the default value.

    Returns
    -------
    comptime : array_like
       List with tuples of target set size and expected computational time for a fixed
       capacity.

    Examples
    --------
    >>> m, targets = ch_parser.parse('NYCstreetnetwork.json')
    >>> comptime = timeit_difftargetsizes(m, cap=100, size_bound=200, obj=BUCHI)

    Notes
    -----
    The values of timings returned might significantly vary depending on the
    machine configuration.

    """

    comptime = []

    targetsize_list = np.linspace(20,size_bound, num_samples, dtype = int)
    targetsize_list = targetsize_list.astype(np.int64)
    for i in range(num_samples):
        def calc_time():
            targets = np.random.randint(0, high=m.num_states, size=(targetsize_list[i]))
            s = energy_solver.EnergySolver(m, cap=cap, targets=targets)
            s.get_strategy(obj, recompute=True)
        comptime.append((targetsize_list[i], timeit.timeit(calc_time, number=num_tests)/num_tests))
    return comptime

def histogram(comptime):
    """
    PLot histogram of the computation time for different targets set of same
    size. Y axis is expressed in terms of pecentage of samples. 
    """

    plt.hist(comptime, weights=np.ones(len(comptime))/len(comptime),color='skyblue')
    plt.xlabel('Computation Time (sec)')
    plt.title('Histogram of Buchi Computational Time. Mean: {} and SD: {}'.format(round(np.mean(comptime),4), round(np.std(comptime),4)))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()

def visualize_strategy(strategy, m, targets):
    """
    Visualize the actions suggested by a strategy for a given MDP modeling an
    AEV in a street network.
    
    Parameters
    ----------
    
    strategy : list
        A list of length equal to the number of states where each element is a
        dict element with keys are energy values and values as the action.
    
    m : mdp;  object of class ConsMDP
       A valid Markov Decision Process.
    
    targets : array_like, list of target states
       A list containing all the target states.
       
    Returns
    -------
    mapobj; folium map object
        A folium map object containing all the states and actions.
    
    Examples
    --------
    >>> m, targets = ch_parser.parse('NYCstreetnetwork.json')
    >>> s = energy_solver.EnergySolver(m, cap=100, targets=targets)
    >>> strategy = s.get_strategy(BUCHI, recompute=True)
    >>> mapobj = visualize_strategy(strategy, m, targets)
    
    """
    # Initialize
    states_nostrategy = [] 
    states_reload = [] 
    states_target = []
    strategy_updated = {}
    map_statelabels = m.names
    
    # Map states to original labels
    for index in range(m.num_states):
        if not map_statelabels[index][:2] == 'ps':
            if strategy[index]:
                strategy_updated.update({map_statelabels[index]: strategy[index]})
            else:
                states_nostrategy.append(map_statelabels[index])

    # Map target and reload states to original labels
    for item in targets:
        states_target.append(map_statelabels[item])
    for index, item in enumerate(m.reloads):
        if item:
            states_reload.append(map_statelabels[index])
            
    # Extract resultant states for actions
    dynamics = {}
    with open('NYCstreetnetwork.json','r') as f:
        raw_data = json.load(f)
        for edge in raw_data["edges"]:
            tail = edge["tail"]
            head = edge["head"]
            if tail[:2] == 'pa':
                action_label = tail[4:]
                dynamics.update({action_label:head})
                
    # Map actions in strategy to resultant states
    for key, value in strategy_updated.items():
        for energy, action_label in value.items():
            value.update({energy: dynamics[action_label]})
    
    mapobj = create_geoplot(states_nostrategy, states_reload, states_target, strategy_updated)   
    return mapobj         

def create_geoplot(states_nostrategy, states_reload, states_target, strategy_updated):
    
    """
    Helper function to visualize the data from MDP and strategy on an OpenStreetMap.
    Uses geodata of the street network stored as a graph in 'nyc.graphml' file.
    """
    
    # Load NYC Geodata
    path = os.path.abspath("nyc.graphml")
    G = nx.MultiDiGraph(nx.read_graphml(path))
    for _, _, data in G.edges(data=True, keys=False):
        data['speed_mean'] = float(data['speed_mean'])
        data['speed_sd'] = float(data['speed_sd'])
        data['time_mean'] = float(data['time_mean'])
        data['time_sd'] = float(data['time_sd'])
        data['energy_levels'] = ast.literal_eval(data['energy_levels'])
    for _, data in G.nodes(data=True):
        data['reload'] = ast.literal_eval(data['reload'])
        data['lat'] = float(data['lat'])
        data['lon'] = float(data['lon'])

    # Create baseline map with edges 
    nodes_all = {}
    for node in G.nodes.data():
        name = str(node[0])
        point = [node[1]['lat'], node[1]['lon']]
        nodes_all[name] = point
    global_lat = []; global_lon = []
    for name, point in nodes_all.items():
        global_lat.append(point[0])
        global_lon.append(point[1])
    min_point = [min(global_lat), min(global_lon)]
    max_point =[max(global_lat), max(global_lon)]
    m = folium.Map(zoom_start=1, tiles='cartodbpositron')
    m.fit_bounds([min_point, max_point])
    for edge in G.edges:
        points = [(G.nodes[edge[0]]['lat'], G.nodes[edge[0]]['lon']),
                  (G.nodes[edge[1]]['lat'], G.nodes[edge[1]]['lon'])]
        folium.PolyLine(locations=points,
                        color='gray',
                        weight=2,
                        opacity=0.8).add_to(m)
    
    for key, value in strategy_updated.items():
        color = '#2f2f2f'
        for energy, end_state in value.items():
            points = [(G.nodes[key]['lat'], G.nodes[key]['lon']),
                      (G.nodes[end_state]['lat'], G.nodes[end_state]['lon'])]
            line = folium.PolyLine(locations=points,
                                   color=color,
                                   tooltip=str(energy),
                                   weight=1.5).add_to(m)
            attr = {'fill': color, 'font-size': '12'}
            plugins.PolyLineTextPath(line,'\u25BA',
                                     repeat=False,
                                     center=True,
                                     offset=3.5,
                                     attributes=attr).add_to(m)
            folium.CircleMarker(location=[G.nodes[key]['lat'], G.nodes[key]['lon']],
                        radius= 2,
                        color=color,
                        fill=True).add_to(m)
    
    # Add reload states, target states, and states with no prescribed action
    nodes_reload = {}
    nodes_target = {}
    nodes_nostrategy = {}
    for node in G.nodes.data():
        if node[0] in states_reload:
            name = str(node[0])
            point = [node[1]['lat'], node[1]['lon']]
            nodes_reload[name] = point
        if node[0] in states_target:
            name = str(node[0])
            point = [node[1]['lat'], node[1]['lon']]
            nodes_target[name] = point
        if node[0] in states_nostrategy:
            name = str(node[0])
            point = [node[1]['lat'], node[1]['lon']]
            nodes_nostrategy[name] = point

    # Plot reload states
    for node_name, node_point in nodes_reload.items():
        folium.CircleMarker(location=[node_point[0], node_point[1]],
                        radius= 3,
                        popup = 'reload state',
                        color="#22af4b",
                        fill_color = "#22af4b",
                        fill_opacity=1,
                        fill=True).add_to(m)
    # Plot target nodes
    for node_name, node_point in nodes_target.items():
        folium.CircleMarker(location=[node_point[0], node_point[1]],
                        radius= 3,
                        popup = 'target state',
                        color="#0f89ca",
                        fill_color = "#0f89ca",
                        fill_opacity=1,
                        fill=True).add_to(m)
    # Plot no strategy nodes
    for node_name, node_point in nodes_nostrategy.items():
        folium.CircleMarker(location=[node_point[0], node_point[1]],
                        radius= 3,
                        popup = 'no guarantees state',
                        color='red',
                        fill_color = 'red',
                        fill_opacity=1,
                        fill=True).add_to(m)
    return m

def extract_data(m, targets):
    
    # Initialize 
    states_reload = [] 
    states_target = []
    map_statelabels = m.names

    # Map target and reload states to original labels
    for item in targets:
        states_target.append(map_statelabels[item])
    for index, item in enumerate(m.reloads):
        if item:
            states_reload.append(map_statelabels[index])
    
    return states_reload, states_target


