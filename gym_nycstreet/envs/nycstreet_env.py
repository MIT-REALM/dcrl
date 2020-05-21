import numpy as np
from gym import Env
from gym.spaces import Box
import sys
sys.path.insert(0, '..')
import gym_nycstreet.fimdp.nyc_parser as nyc_parser
import gym_nycstreet.fimdp.NYCtools as NYCtools
import ast
import networkx as nx
import matplotlib.pyplot as plt
import time


class NYCStreetEnv(Env):

    ENERGY_FULL = 2
    NUM_ANGLES = 8
    MAX_RELOADS = 1000
    REWARD_WEIGHTS = 10
    SLEEP_TIME = 0.5
    ENERGY_LOW_THRES = 0
    MAX_STEP_PER_EPISODE = 200
    MAX_DISTANCE = 0.2

    def __init__(self, training=True):

        nyc_path = 'gym_nycstreet/envs/nyc.graphml'
        G = nx.MultiDiGraph(nx.read_graphml(nyc_path))
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

        nodes_xy = []
        nodes_latlon = []
        node_name_to_index = {}
        node_index_to_name = []
        for i, node in enumerate(G.nodes.data()):
            name = str(node[0])
            node_name_to_index[name] = i
            node_index_to_name.append(name)
            nodes_xy.append([float(node[1]['x']), float(node[1]['y'])])
            nodes_latlon.append([float(node[1]['lat']), float(node[1]['lon'])])
    
        self.nodes_xy = np.array(nodes_xy)
        self.nodes_latlon = np.array(nodes_latlon)
        self.node_name_to_index = node_name_to_index
        self.node_index_to_name = node_index_to_name
        self.num_nodes = len(self.node_index_to_name)
        self.stop_early = False
        self.fig = None
        self.use_render = False
        self.training = training

        self.nodes_xy_vertical = self._rotate_nycstreet(
            nodes_xy - np.mean(nodes_xy, axis=0, keepdims=True))

        self.transitions = []
        for _ in range(self.num_nodes):
            self.transitions.append(
                {'coor': None, 'next': [], 'coor_diff': [], 
                 'reload': False, 'target': False})

        for edge in G.edges:
            i_from = self.node_name_to_index[edge[0]]
            i_to = self.node_name_to_index[edge[1]]
            coor_from = self.nodes_xy_vertical[i_from]
            coor_to = self.nodes_xy_vertical[i_to]
            coor_diff = coor_to - coor_from
            if self.transitions[i_from]['coor'] is not None:
                assert np.all(np.equal(
                    self.transitions[i_from]['coor'], coor_from))
            else:
                self.transitions[i_from]['coor'] = coor_from
            self.transitions[i_from]['next'].append(i_to)
            self.transitions[i_from]['coor_diff'].append(coor_diff)

        path = 'gym_nycstreet/envs/NYCstreetnetwork.json'
        m, targets = nyc_parser.parse(path)
        states_reload, states_target = NYCtools.extract_data(m, targets)
        for i, state in enumerate(states_reload):
            if i == self.MAX_RELOADS:
                break
            self.transitions[self.node_name_to_index[state]]['reload'] = True
        for i, state in enumerate(states_target):
            self.transitions[self.node_name_to_index[state]]['target'] = True

        self.not_target = []
        for i in range(self.num_nodes):
            self.not_target.append(int(not self.transitions[i]['target']))
        self.not_target = np.array(self.not_target)
        self.not_target = self.not_target / np.sum(self.not_target)
        
        for trans in self.transitions:
            trans['next'] = np.array(trans['next'])
            trans['coor_diff'] = np.array(trans['coor_diff'])
            valid_actions = []
            for i in range(self.NUM_ANGLES):
                angle = i * 2 * np.pi / self.NUM_ANGLES
                vector = np.array([np.cos(angle), np.sin(angle)])
                coor_diff = trans['coor_diff']
                dot_product = coor_diff.dot(
                        vector) / (1e-5 + np.linalg.norm(coor_diff, axis=1))
                valid_actions.append(np.max(dot_product) > 0.1)
            assert np.sum(valid_actions) > 0
            trans['valid_actions'] = np.array(valid_actions, dtype=np.float32)

        self.coor_to_reload_index = {}
        i = 0
        for trans in self.transitions:
            if trans['reload']:
                x, y = trans['coor']
                self.coor_to_reload_index[str((x, y))] = i
                i = i + 1
        self.num_reloads = len(self.coor_to_reload_index)

        self.observation_space = Box(
            -np.ones(4) * np.inf, np.ones(4) * np.inf, dtype=np.float64)
        self.action_space = Box(
            np.zeros(self.NUM_ANGLES+1), np.ones(self.NUM_ANGLES+1), dtype=np.float64)

    def reset(self):
        self.s = np.random.choice(np.arange(self.num_nodes), p=self.not_target)
        self.lastaction = None
        self.energy = np.copy(self.ENERGY_FULL)
        self.done = False
        self.num_steps_curr_episode = 0
        obs = self.nodes_xy_vertical[self.s]
        obs = np.concatenate([obs, [self.energy], [0]])
        if self.use_render:
            plt.ion()
            plt.close()
            self._render_init()
        return obs

    def step(self, a):
        """
        - a: array of length NUM_ANGLES + 1 representing the angle
        """
        angle_prob, reload_prob = a[:self.NUM_ANGLES], a[self.NUM_ANGLES]
        a = self._choose_action_from_prob(angle_prob)
        angle = (a + np.random.normal() * 0.25) * 2 * np.pi / self.NUM_ANGLES
        vector = np.array([np.cos(angle), np.sin(angle)])
        coor_diff = self.transitions[self.s]['coor_diff']
        dot_product = coor_diff.dot(
                vector) / (1e-5 + np.linalg.norm(coor_diff, axis=1))
        next_state_i = self.transitions[self.s]['next'][np.argmax(dot_product)]
        energy_consume = np.linalg.norm(
            self.transitions[self.s]['coor_diff'][np.argmax(dot_product)])
        energy_consume = min(self.MAX_DISTANCE, energy_consume)
        self.energy = self.energy - energy_consume
        self.s = next_state_i
        if self.transitions[self.s]['reload']:
            self.energy = min(self.ENERGY_FULL, 
                              self.energy + self.ENERGY_FULL * reload_prob)
            reloaded = reload_prob
        else:
            reloaded = 0
        if self.transitions[self.s]['target']:
            self.done = True
        obs = self.nodes_xy_vertical[self.s]
        obs = np.concatenate([obs, [self.energy], [reloaded]])
        rew = 20 if self.done else -energy_consume
        rew = rew * self.REWARD_WEIGHTS
        if self.num_steps_curr_episode == self.MAX_STEP_PER_EPISODE or \
           self.training and self.energy < self.ENERGY_LOW_THRES:
            self.done = True
        self.num_steps_curr_episode = self.num_steps_curr_episode + 1
        return (obs, rew, self.done, {})

    def render(self, mode):
        if self.fig is None:
            self._render_init()
        plt.scatter(self.nodes_xy_vertical[self.s, 0], 
                    self.nodes_xy_vertical[self.s, 1], 
                    s=50, color='orange', alpha=0.5)
        self.fig.canvas.draw()
        time.sleep(self.SLEEP_TIME)

    def set_use_render(self):
        self.use_render = True

    def _render_init(self):
        targets = []
        reloads = []

        for tran in self.transitions:
            if tran['target']:
                targets.append(tran['coor'])
            if tran['reload']:
                reloads.append(tran['coor'])
        targets = np.array(targets)
        reloads = np.array(reloads)

        fig = plt.figure(figsize=(10, 10))
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.scatter(targets[:, 0], targets[:, 1], 
                    color='blue', s=50, alpha=0.5)
        plt.scatter(reloads[:, 0], reloads[:, 1], 
                    color='red', s=50, alpha=0.5)
        plt.scatter(self.nodes_xy_vertical[:, 0], self.nodes_xy_vertical[:, 1], 
                    s=10, color='black', alpha=0.5)
        self.fig = fig

    def _choose_action_from_prob(self, prob):
        prob = self.transitions[self.s]['valid_actions'] * prob
        prob = np.maximum(prob, 1e-5)
        prob = prob / np.sum(prob)
        a = np.random.choice(np.arange(self.NUM_ANGLES), p=prob)
        return a

    def _rotate_nycstreet(self, p):
        top_index = np.argmax(p[:, 1])
        left_index = np.argmin(p[:, 0])
        diff = p[top_index] - p[left_index]
        rot = np.arctan2(diff[1], diff[0])
        delta_rot = np.pi / 2 - rot
        mat = np.array([[np.cos(delta_rot), np.sin(delta_rot)],
                        [-np.sin(delta_rot), np.cos(delta_rot)]])
        p_rot = p.dot(mat)
        p_rot = p_rot / (1e-5 + np.mean(np.abs(p_rot)))
        return p_rot
