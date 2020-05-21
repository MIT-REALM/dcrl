"""Run this file from within the 'examples' folder:
>> cd examples
>> python ddpg_series_omega_control.py
"""
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, Concatenate
from keras.optimizers import Adam
from keras_agent.agents import DDPGAgent
from keras_agent.memory import SequentialMemory
from keras_agent.random import OrnsteinUhlenbeckProcess
from gym.wrappers import FlattenObservation
from gym.core import Wrapper
import argparse
import sys
import os
import numpy as np
from scipy import interpolate
sys.path.append(os.path.abspath(os.path.join('..')))
import gym_electric_motor as gem
from gym_electric_motor.reference_generators import \
        WienerProcessReferenceGenerator, \
        SquareReferenceGenerator
from gym_electric_motor.visualization import MotorDashboard
from gym.spaces import Box, Tuple
import matplotlib.pyplot as plt


class TemperatureWrapper(Wrapper):
    """
    The following environment appends the normalized temperature to the observation space.
    """
    TEMP_DECAY = 0.9
    I_IDX = 2
    MEMORY_EXPAND = 4
    MEMORY_SIZE = [100, 20]
    BETA = 1.0
    LAGRANGE_GRID_WIDTH = 0.02
    def __init__(self, environment, density_max_func, constrained=True, 
                output_dir='outputs', gamma=0.99, save_traj=False):
        super().__init__(environment)
        self.temperature = 0.0
        self._kernel_width = 0.5
        self._gamma = gamma
        self._pointer = 0
        self._num_updates = 0
        self._output_dir = output_dir
        self._constrained = constrained
        self._reward_history = []
        self._trajectory = []
        self._save_traj = save_traj
        self._stop_early = False
        if not os.path.exists(os.path.join(output_dir, 'visualize')):
            os.makedirs(os.path.join(output_dir, 'visualize'))
        if not os.path.exists(os.path.join(output_dir, 'weights')):
            os.makedirs(os.path.join(output_dir, 'weights'))
        if save_traj and not os.path.exists(os.path.join(output_dir, 'trajectory')):
            os.makedirs(os.path.join(output_dir, 'trajectory'))
        self._density_max_func = density_max_func
        if self._constrained:
            self.observation_space = Tuple((Box(
                np.concatenate((environment.observation_space[0].low, [0.0])),
                np.concatenate((environment.observation_space[0].high, [1.0]))
            ), environment.observation_space[1]))
        else:
            self.observation_space = environment.observation_space
        self._temperature_memory_prev = np.zeros(self.MEMORY_SIZE, dtype=np.float32)
        self._temperature_memory_prev[0, 0] = 1e-9
        self._lagrange_memory_prev = np.zeros(self.MEMORY_SIZE, dtype=np.float32)
        self._lagrange_memory_curr = np.zeros(self.MEMORY_SIZE, dtype=np.float32)
        self._lagrange_interp = None
        self._update_lagrange_interp()

    def step(self, action):
        (state, ref), rew, term, info = self.env.step(action)
        self._reward_history.append(rew)
        i = state[self.I_IDX]
        rew = self._add_temperature_reward(self.temperature, rew)
        self.temperature = self.temperature * self.TEMP_DECAY + (1 - self.TEMP_DECAY) * i**2
        if self._save_traj:
            self._trajectory.append(np.concatenate((state, [self.temperature], ref, action)))
        if self._constrained:
            state = np.concatenate((state, [self.temperature]))
        return (state, ref), rew, term, info

    def reset(self, **kwargs):
        state, ref = self.env.reset()
        if self._save_traj:
            self._trajectory.append(np.concatenate((state, [self.temperature], ref, [0.0])))
        if self._constrained:
            state = np.concatenate((state, [0.0]))
        return state, ref

    def config_save_weights(self, save_weights_funct):
        self._save_weights_funct = save_weights_funct

    def get_history_reward(self, clear=True):
        mean_reward = np.mean(self._reward_history)
        self._reward_history = [] if clear else self._reward_history
        return mean_reward
        
    def _save_weights(self):
        save_path = os.path.join(self._output_dir, 
            'weights/ddpg_dcrl_series_weights_{}.h5f'.format(self._num_updates))
        self._save_weights_funct(save_path, overwrite=True)

    def _save_trajectory(self):
        save_path = os.path.join(self._output_dir, 
            'trajectory/ddpg_dcrl_series_traj_{}.npy'.format(self._num_updates))
        np.save(open(save_path, 'wb'), np.array(self._trajectory))
        self._trajectory = []

    def _add_temperature_reward(self, temperature, reward):
        lagrange_multiplier = self._compute_lagrange(temperature)
        if self._constrained:
            reward = reward + lagrange_multiplier
        column, row = divmod(self._pointer, self.MEMORY_SIZE[0])
        column = column // self.MEMORY_EXPAND
        self._temperature_memory_prev[row, column] = temperature
        self._lagrange_memory_curr[row, column] = lagrange_multiplier
        self._pointer = self._pointer + 1
        if self._pointer == np.prod(self.MEMORY_SIZE) * self.MEMORY_EXPAND:
            self._pointer = 0
            self._lagrange_memory_prev = np.copy(self._lagrange_memory_curr)
            self._update_lagrange_interp()
            self._update_kernel_width()
            self.density_satisfied = self._visualize_density(os.path.join(
                self._output_dir, 'visualize', 
                'density_plot_{}.png'.format(self._num_updates)))
            if self._save_traj:
                self._save_trajectory()
            self._save_weights()
            self._num_updates = self._num_updates + 1
        return reward

    def _compute_lagrange(self, temperature):
        lagrange_prev = self._get_lagrange_prev(temperature)
        density_prev = self._get_density_prev(temperature)
        density_max = self._get_density_max(temperature)
        lagrange = max(-1000, min(0, lagrange_prev - self.BETA * self._piecewise_identity(
            density_prev - density_max)))
        return lagrange

    def _piecewise_identity(self, x, alpha=-20, beta=0):
        if x < alpha:
            return (x - alpha) * 1e-6
        elif x < beta:
            return 0
        else:
            return (x - beta) * 10

    def _get_lagrange_prev(self, temperature):
        return self._lagrange_interp(temperature)

    def _update_lagrange_interp(self):
        t = self._temperature_memory_prev.flatten()
        l = self._lagrange_memory_prev.flatten()
        xs = np.linspace(0, 1, num=100)
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
        return

    def _update_kernel_width(self):
        self._kernel_width = max(1e-3, np.std(self._temperature_memory_prev.flatten()) / 2.0)
        return

    def _get_density_prev(self, temperature):
        k = 1 - ((temperature - self._temperature_memory_prev) / self._kernel_width)**2
        k = k * 3.0 / (4 * self._kernel_width)
        gamma_pow = np.power(self._gamma, np.arange(k.shape[0]))
        density = np.mean(gamma_pow.dot(np.maximum(k, 0)))
        return density

    def _get_density_max(self, temperature):
        return self._density_max_func(temperature)

    def _visualize_density(self, save_path='density_plot.png'):
        xs = np.linspace(0.0, 1.0, 100)
        density = np.array([self._get_density_prev(x) for x in xs])
        density_max = np.array([self._get_density_max(x) for x in xs])
        density_satisfied = np.all(np.less_equal(density, density_max))
        lagrange = np.array([self._compute_lagrange(x) for x in xs])
        npz_path = save_path[:-4] + '.npz'
        np.savez(npz_path, xs=xs, density=density, density_max=density_max, lagrange=lagrange)
        plt.figure()
        plt.subplot(211)
        plt.plot(xs, density, label='Density')
        start_index = np.max((density_max > 200) * np.arange(len(density_max)))
        plt.plot(xs[start_index+1:], density_max[start_index+1:], 
                 label='Maximum Allowed Density', color='darkred')
        plt.xlabel('Normalized Relative Temperature')
        plt.legend()
        plt.subplot(212)
        plt.plot(xs, lagrange, label='Lagrange Multiplier')
        plt.xlabel('Normalized Relative Temperature')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return density_satisfied

    @property
    def stop_early(self):
        return self._stop_early


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--output", type=str, default='outputs')
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--constrained", type=int, default=1)
    args = parser.parse_args()
    # Create the environment
    # Default DcSeries Motor Parameters are changed to have more dynamic system and to see faster learning results.
    env = gem.make(
        'emotor-dc-series-cont-v1',
        # Pass a class with extra parameters
        visualization=MotorDashboard, visu_period=5,
        motor_parameter=dict(r_a=2.5, r_e=4.5, l_a=9.7e-3, l_e_prime=9.2e-3, l_e=9.2e-3, j_rotor=0.001),
        # Take standard class and pass parameters (Load)
        load_parameter=dict(a=0, b=.0, c=0.01, j_load=.001),
        reward_weights={'omega': 1000},
        reward_power=0.5,
        observed_states=None,  # Constraint violation monitoring is disabled for presentation purpose
        # Pass a string (with extra parameters)
        plotted_variables=['omega', 'i', 'u'],
        ode_solver='scipy.solve_ivp', solver_kwargs=dict(method='BDF'),
        # Pass an instance
        reference_generator=SquareReferenceGenerator(reference_state='omega')
        # reference_generator=WienerProcessReferenceGenerator(reference_state='omega', sigma_range=(5e-3, 1e-2))
    )

    density_max_func = interpolate.interp1d(
        x=[0.15, 0.2, 0.25, 0.3, 1.0],
        y=[120, 80, 10, 10, 10],
        kind='slinear',
        fill_value=10000,
        bounds_error=False,
    )
    
    env = TemperatureWrapper(
        env, density_max_func, args.constrained, gamma=0.95, output_dir=args.output,
        save_traj=True
    )

    # Keras-rl DDPG-agent accepts flat observations only
    env = FlattenObservation(env)
    nb_actions = env.action_space.shape[0]

    window_length = 1
    actor = Sequential()
    actor.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
    actor.add(Dense(16, activation='relu'))
    actor.add(Dense(16, activation='relu'))
    actor.add(Dense(nb_actions, activation='sigmoid'))
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(window_length,) + env.observation_space.shape, name='observation_input')
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
        memory_size=env.MEMORY_SIZE + [env.MEMORY_EXPAND],
        window_length=window_length
    )

    # Create a random process for exploration during training
    random_process = OrnsteinUhlenbeckProcess(
        theta=0.5,
        mu=0.0,
        sigma=0.2
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
    
    # Start training for 7.5M simulation steps (1.5M training steps with actions repeated 5 times)

    env.config_save_weights(agent.save_weights)
    if args.mode == 'train':
        if args.weights:
            agent.load_weights(args.weights)
        agent.fit(
            env,
            nb_steps=800000,
            visualize=False,
            action_repetition=1,
            verbose=1,
            nb_max_start_steps=0,
            nb_max_episode_steps=8000,
            log_interval=8000,
            callbacks=[]
        )
        agent.save_weights(os.path.join(
            args.output, 'weights/ddpg_dcrl_series_weights.h5f'), overwrite=True)
    
    elif args.mode == 'test':
        agent.load_weights(args.weights)
        # Test the agent
        hist = agent.test(
            env,
            nb_episodes=10,
            action_repetition=1,
            nb_max_episode_steps=100000,
            visualize=True
        )

    elif args.mode == 'test_all':
        results = []
        for i in range(1000):
            if not os.path.exists(os.path.join(args.weights, 
                'ddpg_dcrl_series_weights_{}_actor.h5f'.format(i))):
                continue
            agent.load_weights(os.path.join(args.weights, 
                'ddpg_dcrl_series_weights_{}.h5f'.format(i)))
            # Test the agent
            agent.test(
                env,
                nb_episodes=1,
                action_repetition=1,
                nb_max_episode_steps=8000,
                visualize=False
                )
            mean_reward = env.get_history_reward()
            density_satisfied = env.density_satisfied
            results.append([i, mean_reward, density_satisfied])
            i = i + 1
            
        results = np.array(results)
        results_filt = results[results[:, 2] > 0]
        argsort = np.argsort(-results_filt[:, 1])
        print('|     index     |     reward     |  satisfied  |')
        print(results_filt[argsort])

        argsort = np.argsort(-results[:, 1])
        print('|     index     |     reward     |  satisfied  |')
        print(results[argsort])
