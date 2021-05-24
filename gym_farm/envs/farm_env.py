import numpy as np
from gym import Env
from gym.spaces import Box
import sys
sys.path.insert(0, '..')
import matplotlib.pyplot as plt
import os
import time


class FarmEnv(Env):
    
    SLEEP_TIME = 0
    VELOCITY_DECAY = 0.6
    ACCELERATION_DECAY = 0.6
    GRID_SIZE = 50
    DRONE_DENSITY_DEMAND = [0, 1, 0, 0, 1, 1]
    DRONE_DENSITY_DEMAND_DELTA = 1
    STEPS_PER_EPISODE = 20000
    MAX_SPEED = 5
    MAX_INTENSITY = 3
    VELOCITY_GRID_NUM = 100

    def __init__(self, training=True, constrained=1, output_dir='output'):
        self.training = training
        self.constrained = constrained
        self.output_dir = output_dir
        self.use_render = False
        self.stop_early = False
        self.s = np.zeros(2, dtype=np.float32)
        self.v = 0
        self.fig = None
        self.grid_x, self.grid_y = np.meshgrid(
            np.arange(self.GRID_SIZE), np.arange(self.GRID_SIZE),
            indexing='ij')
        self.observation_space = Box(-np.ones(2) * self.GRID_SIZE, 
                                      np.ones(2) * self.GRID_SIZE)
        self.action_space = Box(np.zeros(1), np.ones(1))
        self.farm_mask = np.load(open('gym_farm/envs/farm_mask.npy', 'rb'))
        self.drone_density_min = np.zeros(
            np.shape(self.farm_mask), dtype=np.float32)
        self.drone_density_max = np.zeros(
            np.shape(self.farm_mask), dtype=np.float32)
        for i, demand in enumerate(self.DRONE_DENSITY_DEMAND):
            self.drone_density_min[np.where(self.farm_mask == i)] = demand
            max_demand = demand + self.DRONE_DENSITY_DEMAND_DELTA
            self.drone_density_max[np.where(self.farm_mask == i)] = max_demand
        self.num_updates = 0
        self.drone_density = np.zeros(
            shape=[self.GRID_SIZE, self.GRID_SIZE], dtype=np.float32)
        self.drone_density_prev = np.zeros(
            shape=[self.GRID_SIZE, self.GRID_SIZE], dtype=np.float32)
        self.drone_lagrange_min_prev = np.zeros(
            shape=[self.GRID_SIZE, self.GRID_SIZE], dtype=np.float32)
        self.drone_lagrange_max_prev = np.zeros(
            shape=[self.GRID_SIZE, self.GRID_SIZE], dtype=np.float32)
        self.velocity_density_prev = np.zeros(
            shape=self.VELOCITY_GRID_NUM, dtype=np.float32)
        self.velocity_density = np.zeros(
            shape=self.VELOCITY_GRID_NUM, dtype=np.float32)
        self.velocity_lagrange_prev = np.zeros(
            shape=self.VELOCITY_GRID_NUM, dtype=np.float32)
        self.rcpo_lagrange = 0

    def reset(self):
        self.drone_density = np.zeros([self.GRID_SIZE, self.GRID_SIZE], np.float32)
        self.s = np.zeros(2, dtype=np.float32)
        self.v = 0
        self.done = False
        state = np.array([self.v, self._future_drone_density_min()])
        return state

    def step(self, a):
        acceleration = max(0.01, float(a)) * self.ACCELERATION_DECAY
        self.v = self.VELOCITY_DECAY * self.v + (
            1 - self.VELOCITY_DECAY) * acceleration * self.MAX_SPEED
        if np.mod(self.s[1], 2) == 0:
            self.s[0] = min(self.GRID_SIZE - 1, self.s[0] + self.v)
            if self.s[0] == self.GRID_SIZE - 1:
                self.s[1] = self.s[1] + 1
        else:
            self.s[0] = max(0, self.s[0] - self.v)
            if self.s[0] == 0:
                self.s[1] = self.s[1] + 1
        self.s = np.clip(self.s, 0, self.GRID_SIZE - 1)
        state = np.array([self.v, self._future_drone_density_min()])
        rew = -0.1 / (1 + self.v)
        rew = self._add_drone_density_reward(self.s, rew)
        rew = self._add_velocity_density_reward(self.v, rew)
        self._update_drone_density(intensity=1)
        self._update_velocity_density(self.v)
        if np.all(self.s == self.GRID_SIZE - 1):
            self._update_drone_lagrange()
            self._update_velocity_lagrange()
            self._visualize_drone_density(
                os.path.join(self.output_dir, 'visualize', 
                'drone_density_plot_{}.png'.format(self.num_updates)))
            self.done = True
            self._save_weights()
            self.num_updates = self.num_updates + 1
        return (state, rew, self.done, {})

    def render(self, mode):
        if not self.use_render:
            return
        if self.fig is None:
            self._render_init()
        # Draw
        plt.pcolor(self.grid_y, self.grid_x, self.drone_density,
                   cmap=plt.get_cmap('GnBu'), vmin=0, vmax=2, alpha=1)
        plt.scatter(self.s[0], self.s[1], s=10, color='darkred')
        self.fig.canvas.draw()
        time.sleep(self.SLEEP_TIME)

    def config_save_weights(self, save_weights_funct):
        self._save_weights_funct = save_weights_funct

    def config_velocity_density_max_funct(self, funct):
        self._velocity_density_max_funct = funct

    def _future_drone_density_min(self):
        x, y = self.s.astype(np.int8)
        if np.mod(y, 2) == 0:
            x_min = int(x)
            x_max = int(min(self.GRID_SIZE, x + self.MAX_SPEED))
        else:
            x_min = int(max(0, x - self.MAX_SPEED))
            x_max = int(x)
        if len(self.drone_density_min[int(y), x_min:x_max]) > 0:
            return np.max(self.drone_density_min[int(y), x_min:x_max])
        return 0

    def _save_weights(self):
        save_path = os.path.join(self.output_dir, 
            'weights/ddpg_farm_{}.h5f'.format(self.num_updates))
        self._save_weights_funct(save_path, overwrite=True)

    def set_use_render(self):
        self.use_render = True

    def _render_init(self):
        plt.ion()
        fig = plt.figure(figsize=(3, 3))
        plt.xlim(0, self.GRID_SIZE)
        plt.ylim(self.GRID_SIZE, 0)
        plt.axis('off')
        self.fig = fig

    def _update_drone_density(self, intensity):
        x, y = self.s
        x_min, x_max = self._get_past_range(x, y)
        area = x_max - x_min + 1
        intensity_abs = intensity * self.MAX_INTENSITY
        self.drone_density[int(y), x_min:x_max] += intensity_abs / area

    def _get_past_range(self, x, y):
        if np.mod(y, 2) == 0:
            x_max = int(min(self.GRID_SIZE - 1, x))
            x_min = int(min(x_max, max(0, x - self.v)))
        else:
            x_max = int(min(self.GRID_SIZE - 1, x + self.v))
            x_min = int(min(x_max, max(0, x)))
        return x_min, x_max

    def _add_drone_density_reward(self, s, reward):
        if self.constrained == 0 or not self.training:
            return reward
        x, y = s
        x_min, x_max = self._get_past_range(x, y)
        density = self.drone_density[int(y), x_min:x_max+1]
        if density.shape[0] == 0:
            return reward
        density_min = self.drone_density_min[int(y), x_min:x_max+1]
        density_max = self.drone_density_max[int(y), x_min:x_max+1]
        density_reward = np.mean(density_min) + np.mean(
            np.minimum(0, density - density_min))
        density_cost = np.mean(np.minimum(0, density_max - density))
        lagrange_min = self.drone_lagrange_min_prev[int(y), int(x)]
        lagrange_max = self.drone_lagrange_max_prev[int(y), int(x)]
        if self.constrained == 1:
            if np.sum(density_min) == 0:
                return reward + density_cost * lagrange_max
            return reward + density_reward * lagrange_min + density_cost * lagrange_max
        else:
            raise NotImplementedError

    def _add_velocity_density_reward(self, velocity, reward):
        if self.constrained == 0 or not self.training:
            return reward
        elif self.constrained == 1:
            velocity_index = np.clip(
                int(velocity * self.VELOCITY_GRID_NUM / self.MAX_SPEED),
                0, self.VELOCITY_GRID_NUM - 1)
            return reward + self.velocity_lagrange_prev[velocity_index]
        else:
            raise NotImplementedError

    def _rcpo_velocity_reward(self, velocity, thres=0.4, alpha=0.01):
        thres = thres * self.MAX_SPEED
        self.rcpo_lagrange = min(10,
            self.rcpo_lagrange + alpha * max(0, velocity - thres))
        return -max(0, velocity - thres) * self.rcpo_lagrange

    def _update_drone_lagrange(self, alpha=1):
        self.drone_density_prev = (self.drone_density_prev + self.drone_density) / 2
        self.drone_lagrange_min_prev = np.minimum(20, np.maximum(1, 
            self.drone_lagrange_min_prev + alpha * (
                self.drone_density_min - self.drone_density_prev)))
        self.drone_lagrange_max_prev = np.minimum(20, np.maximum(1, 
            self.drone_lagrange_max_prev + alpha * (
                self.drone_density_prev - self.drone_density_max)))

    def _update_velocity_density(self, velocity):
        velocity_index = np.clip(
                int(velocity * self.VELOCITY_GRID_NUM / self.MAX_SPEED),
                0, self.VELOCITY_GRID_NUM - 1)
        self.velocity_density[int(velocity_index)] += 1

    def _update_velocity_lagrange(self, alpha=100):
        self.velocity_density_prev = (
            self.velocity_density_prev + self.velocity_density) / 2
        self.velocity = np.zeros(
            shape=self.VELOCITY_GRID_NUM, dtype=np.float32)
        velocity_density_norm = self.velocity_density_prev / np.sum(
            self.velocity_density_prev)
        self.velocity_lagrange_prev = np.maximum(-10, np.minimum(
            0, self.velocity_lagrange_prev - alpha * (
                velocity_density_norm - self._velocity_density_max())))

    def _velocity_density_max(self, x=None):
        if x is not None:
            return self._velocity_density_max_funct(x)
        xs = np.linspace(0, 1, self.VELOCITY_GRID_NUM)
        return self._velocity_density_max_funct(xs)

    def _visualize_drone_density(self, save_path='density_plot.png'):
        drone_density_sat = np.mean(np.logical_and(
            np.greater_equal(self.drone_density_prev, self.drone_density_min),
            np.greater_equal(self.drone_density_max, self.drone_density_prev)))
        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(131)
        ax.pcolor(self.grid_y, self.grid_x, self.drone_density_prev,
                  cmap=plt.get_cmap('GnBu'), vmin=0, vmax=2)
        ax.set_title('Satisfied: {:.4f}'.format(drone_density_sat))
        ax.set_xlim(0, self.GRID_SIZE)
        ax.set_ylim(self.GRID_SIZE, 0)
        ax.set_axis_off()
        ax = fig.add_subplot(132)
        xs = np.linspace(0, 1, self.VELOCITY_GRID_NUM)
        velocity_density_norm = self.velocity_density_prev / np.sum(
            self.velocity_density_prev)
        ax.plot(xs, velocity_density_norm)
        ax.plot(xs, self._velocity_density_max(), color='darkred')
        ax.set_ylim(-0.05, 0.2)
        ax.set_axis_off()
        ax = fig.add_subplot(133)
        ax.plot(xs, self.velocity_lagrange_prev)
        ax.set_ylim(-20, 1)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        npz_path = save_path[:-4] + '.npz'
        np.savez(npz_path, xs=xs, velocity_density=velocity_density_norm, 
                 velocity_density_max=self._velocity_density_max(), 
                 drone_density=self.drone_density_prev,
                 drone_density_min=self.drone_density_min)
