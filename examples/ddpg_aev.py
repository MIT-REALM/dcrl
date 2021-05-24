from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, Concatenate
from keras.optimizers import Adam
from keras_agent.agents import DDPGAgent
from keras_agent.memory import SequentialMemory
from keras_agent.random import GaussianWhiteNoiseProcess
import gym_nycstreet
import gym
from gym.core import Wrapper
import argparse
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join('..')))
import matplotlib.pyplot as plt
from scipy import interpolate


def init_output_folders(output='output'):
    if not os.path.exists(os.path.join(output, 'weights')):
        os.makedirs(os.path.join(output, 'weights'))
    if not os.path.exists(os.path.join(output, 'visualize')):
        os.makedirs(os.path.join(output, 'visualize'))


class EnergyWrapper(Wrapper):
    MEMORY_EXPAND = 5
    MEMORY_SIZE = [100, 20]
    BETA = 0.1
    LAGRANGE_GRID_WIDTH = 0.04
    def __init__(self, environment, density_max_func, reload_density_max=0.005, 
                 constrained=1, output_dir='output', gamma=0.99, save_traj=False):
        super().__init__(environment)
        self.temperature = 0.0
        self._kernel_width = 0.5
        self._gamma = gamma
        self._pointer = 0
        self._pointer_reload = 0
        self._num_updates = 0
        self._num_updates_upload = 0
        self._output_dir = output_dir
        self._constrained = constrained
        self._reward_history = []
        self._save_traj = save_traj
        self._stop_early = False
        self._total_energy = 0
        init_output_folders(output_dir)

        # Placeholders for the energy density constraints
        self._density_max_func = density_max_func
        self._energy_memory_prev = np.zeros(self.MEMORY_SIZE, dtype=np.float32)
        self._energy_memory_prev[0, 0] = 1e-9
        self._lagrange_memory_prev = np.zeros(self.MEMORY_SIZE, dtype=np.float32)
        self._lagrange_memory_curr = np.zeros(self.MEMORY_SIZE, dtype=np.float32)
        self._lagrange_interp = None
        self._update_lagrange_interp()

        # Placeholders for the vehicle density constraints at reload stations
        self._reload_density_max = reload_density_max
        self._reload_memory_prev = np.zeros(self.num_reloads, dtype=np.float32)
        self._reload_memory_curr = np.zeros(self.num_reloads, dtype=np.float32)
        self._reload_lagrange_prev = np.zeros(self.num_reloads, dtype=np.float32)

    def step(self, action):
        state, rew, term, info = self.env.step(action)
        x, y, energy, reloaded = state
        self._reward_history.append(rew)
        rew = self._add_energy_reward(energy, rew)
        rew = self._add_reload_reward(x, y, reloaded, rew)
        self._total_energy = self._total_energy + reloaded
        return state, rew, term, info

    def reset(self, **kwargs):
        state = self.env.reset()
        return state

    def config_save_weights(self, save_weights_funct):
        self._save_weights_funct = save_weights_funct

    def get_history_reward(self, clear=True):
        mean_reward = np.mean(self._reward_history)
        self._reward_history = [] if clear else self._reward_history
        return mean_reward
        
    def _save_weights(self):
        save_path = os.path.join(self._output_dir, 
            'weights/ddpg_aev_{}.h5f'.format(self._num_updates))
        self._save_weights_funct(save_path, overwrite=True)

    def _rcpo_reward_function(self, energy):
        """
        For the reward constrained policy optimization (RCPO) algorithm
        in vehicle energy constraint tasks.
        """
        return min(energy, 0) * 100

    def _rcpo_reload_reward_function(self, x, y, reloaded):
        if not reloaded:
            return 0
        reload_index = self.coor_to_reload_index[str((x, y))]
        density = self._get_reload_density(reload_index)
        density_max = self._reload_density_max
        return min(density_max - density, 0) * 100 * reloaded

    def _add_energy_reward(self, energy, reward):
        """
        Add Lagrange multipliers to rewards in order to satisfy the energy
        constraints based on the DCRL algorithm.
        """ 
        lagrange_multiplier = self._compute_lagrange(energy)
        if not self.training:
            pass 
        elif self._constrained == 1:
            reward = reward + lagrange_multiplier
        elif self._constrained == 0:
            pass
        else:
            raise NotImplementedError
        column, row = divmod(self._pointer, self.MEMORY_SIZE[0])
        column = column // self.MEMORY_EXPAND
        self._energy_memory_prev[row, column] = energy
        self._lagrange_memory_curr[row, column] = lagrange_multiplier
        self._pointer = self._pointer + 1
        if self._pointer == np.prod(self.MEMORY_SIZE) * self.MEMORY_EXPAND:
            self._pointer = 0
            self._lagrange_memory_prev = np.copy(self._lagrange_memory_curr)
            self._update_lagrange_interp()
            self._update_kernel_width()
            self.density_satisfied = self._visualize_energy_density(
                os.path.join(self._output_dir, 'visualize', 
                'energy_density_plot_{}.png'.format(self._num_updates)))
            self._save_weights()
            if not self.training:
                print('Total energy: {:.3f}'.format(self._total_energy))
            self._total_energy = 0
            self._num_updates = self._num_updates + 1
        return reward

    def _compute_lagrange(self, energy):
        lagrange_prev = self._get_lagrange_prev(energy)
        density_prev = self._get_density_prev(energy)
        density_max = self._get_density_max(energy)
        lagrange = max(-5, min(0, lagrange_prev - self.BETA * \
                       self._piecewise_identity(density_prev - density_max, low=-2)))
        return lagrange

    def _get_lagrange_prev(self, energy):
        return self._lagrange_interp(energy)

    def _update_lagrange_interp(self):
        t = self._energy_memory_prev.flatten()
        l = self._lagrange_memory_prev.flatten()
        xs = np.linspace(0, 3, num=1000)
        ys = []
        for x in xs:
            mask = np.logical_and(
                t >= x - self.LAGRANGE_GRID_WIDTH,
                t <= x + self.LAGRANGE_GRID_WIDTH).astype(np.float32)
            if np.sum(mask) > 0:
                ys.append(mask.dot(l) / (np.sum(mask) + 1e-9))
            elif self._lagrange_interp is not None:
                ys.append(self._get_lagrange_prev(x))
            else:
                ys.append(0)
        self._lagrange_interp = interpolate.interp1d(
            xs, ys, bounds_error=False, fill_value=(0, 0))

    def _update_kernel_width(self):
        self._kernel_width = max(0.01, 
            np.std(self._energy_memory_prev.flatten()) / 2.0)

    def _get_density_prev(self, energy):
        k = 1 - ((energy - self._energy_memory_prev) / self._kernel_width)**2
        k = k * 3.0 / (4 * self._kernel_width)
        gamma_pow = np.power(self._gamma, np.arange(k.shape[0]))
        density = np.mean(gamma_pow.dot(np.maximum(k, 0)))
        return density

    def _get_density_max(self, energy):
        return self._density_max_func(energy)

    def _add_reload_reward(self, x, y, reloaded, reward):
        self._pointer_reload = self._pointer_reload + 1
        if self._pointer_reload == np.prod(self.MEMORY_SIZE) * self.MEMORY_EXPAND:
            self._pointer_reload = 0
            self._update_reload_density()
            self._update_reload_lagrange()
            plot_path = os.path.join(self._output_dir, 'visualize', 
                'reload_density_plot_{}.png'.format(self._num_updates_upload))
            self.reload_density_satisfied = self._visualize_reload_density(plot_path)
            self._num_updates_upload += 1
        if not reloaded:
            return reward
        reload_index = self.coor_to_reload_index[str((x, y))]
        self._reload_memory_curr[reload_index] += reloaded
        if self.training:
            if self._constrained == 1:
                reload_lagrange = self._reload_lagrange_prev[reload_index]
                reward = reward + reload_lagrange * reloaded
        return reward

    def _update_reload_density(self, update_ratio=0.5):
        self._reload_memory_prev = self._reload_memory_curr * update_ratio + \
                                   self._reload_memory_prev * (1 - update_ratio)
        self._reload_memory_curr = np.zeros(self.num_reloads, dtype=np.float32)

    def _update_reload_lagrange(self, alpha=1e+2):
        density = self._get_reload_density()
        density_max = self._reload_density_max
        self._reload_lagrange_prev = np.maximum(-10, np.minimum(
            0, self._reload_lagrange_prev - alpha * self._piecewise_identity(
                density - density_max)))

    def _piecewise_identity(self, x, low=-0.002, high=0):
        x_low = x * np.less_equal(x, low)
        x_high = x * np.greater_equal(x, high)
        x = np.minimum(x_low - low, 0) * 0.1 + x_high
        return x

    def _get_reload_density(self, index=None):
        if index is None:
            return self._reload_memory_prev / (
                np.prod(self.MEMORY_SIZE) * self.MEMORY_EXPAND)
        return self._reload_memory_prev[index] / (
                np.prod(self.MEMORY_SIZE) * self.MEMORY_EXPAND)

    def _visualize_energy_density(self, save_path='density_plot.png'):
        xs = np.linspace(0, 3, 1000)
        density = np.array([self._get_density_prev(x) for x in xs])
        density_max = np.array([self._get_density_max(x) for x in xs])
        density_satisfied = np.all(np.less_equal(density, density_max))
        lagrange = np.array([self._compute_lagrange(x) for x in xs])
        npz_path = save_path[:-4] + '.npz'
        np.savez(npz_path, xs=xs, density=density, 
                 density_max=density_max, lagrange=lagrange)
        plt.figure()
        plt.subplot(211)
        plt.plot(xs, density, label='Density')
        plt.plot(xs, density_max, 
                 label='Maximum allowed density', color='darkred')
        plt.xlabel('Energy')
        plt.legend()
        plt.subplot(212)
        plt.plot(xs, lagrange, label='Lagrange multiplier')
        plt.xlabel('Energy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return density_satisfied

    def _visualize_reload_density(self, save_path='reload_density_plot.png'):
        xs = np.arange(self.num_reloads)
        density = self._get_reload_density()
        density_max = np.ones(self.num_reloads) * self._reload_density_max
        density_satisfied = np.all(np.less_equal(density, density_max))
        lagrange = self._reload_lagrange_prev
        npz_path = save_path[:-4] + '.npz'
        np.savez(npz_path, xs=xs, density=density, 
                 density_max=density_max, lagrange=lagrange)
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.bar(xs, density, 0.5, color='#d2c295', label='Density',
               alpha=1.0, edgecolor='grey')
        ax.plot(xs, density_max, color='#8c1515', label='Maximum allowed density')
        ax.set_xlabel('Reload station index')
        plt.legend()
        ax = fig.add_subplot(212)
        ax.bar(xs, lagrange, 0.5, color='#d2c295', label='Lagrange multiplier',
               alpha=1.0, edgecolor='grey')
        ax.set_xlabel('Reload station index')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return density_satisfied


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--output", type=str, default='output')
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--constrained", type=int, default=1)
    args = parser.parse_args()

    init_output_folders(args.output)

    # Create the environment
    env = gym.make('NYCStreet-v0', training = args.mode == 'train')

    def density_max_func(x):
        y = 40 / (1 + np.exp(-(x-0.5)*10))
        return y

    env = EnergyWrapper(
        environment=env, density_max_func=density_max_func, 
        constrained=args.constrained, gamma=0.95, output_dir=args.output
    )

    nb_actions = env.action_space.shape[0]
    window_length = 1
    actor = Sequential()
    actor.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
    actor.add(Dense(32, activation='relu'))
    actor.add(Dense(32, activation='relu'))
    actor.add(Dense(nb_actions, activation='sigmoid'))
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(window_length,) + env.observation_space.shape, 
                              name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(1, activation='linear')(x)
    critic = Model(inputs=(action_input, observation_input), outputs=x)
    print(critic.summary())

    # Create a replay memory
    memory = SequentialMemory(
        limit=10000,
        memory_size=10000,
        window_length=window_length
    )

    # Create a random process for exploration during training
    random_process = GaussianWhiteNoiseProcess(
        mu=0.0,
        sigma=0.3,
        size=nb_actions
    )

    # Create the agent
    agent = DDPGAgent(
        nb_actions=nb_actions,
        actor=actor,
        critic=critic,
        critic_action_input=action_input,
        memory=memory,
        random_process=random_process,
        nb_steps_warmup_actor=2048,
        nb_steps_warmup_critic=1024,
        target_model_update=1000,
        gamma=0.95,
        batch_size=128,
        memory_interval=1
    )
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    env.config_save_weights(agent.save_weights)
    if args.mode == 'train':
        if args.weights:
            agent.load_weights(args.weights)
        agent.fit(
            env,
            nb_steps=1000000,
            visualize=False,
            action_repetition=1,
            verbose=1,
            nb_max_start_steps=0,
            nb_max_episode_steps=100,
            log_interval=10000,
            callbacks=[]
        )
        agent.save_weights(os.path.join(
            args.output, 'weights/ddpg_aev.h5f'), overwrite=True)
    
    elif args.mode == 'test':
        # env.set_use_render()
        agent.load_weights(args.weights)
        # Test the agent
        hist = agent.test(
            env,
            nb_episodes=2000,
            action_repetition=1,
            nb_max_episode_steps=100,
            visualize=False
        )

    elif args.mode == 'test_all':
        for i in range(100):
            if not os.path.exists(os.path.join(args.weights, 
                'ddpg_aev_{}_actor.h5f'.format(i))):
                continue
            agent.load_weights(os.path.join(args.weights, 
                'ddpg_aev_{}.h5f'.format(i)))
            # Test the agent
            hist = agent.test(
                env,
                nb_episodes=500,
                action_repetition=1,
                nb_max_episode_steps=100,
                visualize=False
            )
