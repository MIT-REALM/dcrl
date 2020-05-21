import keras
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, Concatenate
from keras.optimizers import Adam
from keras_agent.agents import DDPGAgent
from keras_agent.memory import SequentialMemory
from keras_agent.random import GaussianWhiteNoiseProcess, OrnsteinUhlenbeckProcess
import gym_farm
import gym
from gym.core import Wrapper
import argparse
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join('..')))
import matplotlib.pyplot as plt
from scipy import interpolate
from keras import layers
from keras import backend as K
from keras.regularizers import l2


class VariationalLayer(layers.Layer):

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        return inputs + tf.random.normal(shape=tf.shape(inputs)) * 0.3


def init_output_folders(output='output'):
    if not os.path.exists(os.path.join(output, 'weights')):
        os.makedirs(os.path.join(output, 'weights'))
    if not os.path.exists(os.path.join(output, 'visualize')):
        os.makedirs(os.path.join(output, 'visualize'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--output", type=str, default='output')
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--constrained", type=int, default=1)
    args = parser.parse_args()

    init_output_folders(args.output)

    # Create the environment
    training = args.mode == 'train'
    env = gym.make(
        'Farm-v0', 
        training=training, 
        output_dir=args.output,
        constrained=args.constrained)

    def density_max_funct(x):
        return 0.05 * np.exp(-(x-0.2) * 9)

    env.config_velocity_density_max_funct(density_max_funct)

    nb_actions = env.action_space.shape[0]
    window_length = 1
    weight_decay = 1e-4

    observation_input_actor = Input(
        shape=(window_length,) + env.observation_space.shape, 
        name='observation_input_actor')
    x = Flatten()(observation_input_actor)
    x = Dense(
        32, 
        activation='relu',
        kernel_regularizer=l2(weight_decay), 
        bias_regularizer=l2(weight_decay))(x)
    x = Dense(
        32, 
        activation='relu',
        kernel_regularizer=l2(weight_decay), 
        bias_regularizer=l2(weight_decay))(x)
    x = Dense(
        nb_actions, 
        activation='sigmoid',
        kernel_regularizer=l2(weight_decay), 
        bias_regularizer=l2(weight_decay))(x)
    actor = Model(inputs=observation_input_actor, outputs=x)
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(window_length,) + env.observation_space.shape, 
                              name='observation_input')
    flattened_observation = Flatten()(observation_input)
    
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(
        32, 
        activation='relu',
        kernel_regularizer=l2(weight_decay), 
        bias_regularizer=l2(weight_decay))(x)
    x = Dense(
        32, 
        activation='relu',
        kernel_regularizer=l2(weight_decay), 
        bias_regularizer=l2(weight_decay))(x)
    x = Dense(
        32, 
        activation='relu',
        kernel_regularizer=l2(weight_decay), 
        bias_regularizer=l2(weight_decay))(x)
    x = Dense(
        1, 
        activation='linear',
        kernel_regularizer=l2(weight_decay), 
        bias_regularizer=l2(weight_decay))(x)
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
        nb_steps_warmup_actor=2000,
        nb_steps_warmup_critic=2000,
        target_model_update=1000,
        custom_model_objects={'VariationalLayer': VariationalLayer},
        gamma=0.99,
        batch_size=128,
        memory_interval=1
    )
    agent.compile(Adam(lr=1e-3, clipnorm=1, decay=1e-5), metrics=['mae'])

    env.config_save_weights(agent.save_weights)
    if args.mode == 'train':
        if args.weights:
            agent.load_weights(args.weights)
        agent.fit(
            env,
            nb_steps=200000,
            visualize=False,
            action_repetition=1,
            verbose=1,
            nb_max_start_steps=0,
            nb_max_episode_steps=env.STEPS_PER_EPISODE,
            log_interval=10000,
            callbacks=[]
        )
        agent.save_weights(os.path.join(
            args.output, 'weights/ddpg_farm.h5f'), overwrite=True)
    
    elif args.mode == 'test':
        # env.set_use_render()
        agent.load_weights(args.weights)
        # Test the agent
        env.set_use_render()
        hist = agent.test(
            env,
            nb_episodes=2000,
            action_repetition=1,
            nb_max_episode_steps=env.STEPS_PER_EPISODE,
            visualize=False
        )

    elif args.mode == 'test_all':
        num_to_test = 20
        counter = 0
        for i in range(100, 20, -1):
            if counter == num_to_test:
                break
            if not os.path.exists(
                os.path.join(args.weights, 'ddpg_farm_{}_actor.h5f'.format(i))):
                continue
            agent.load_weights(os.path.join(args.weights, 
                'ddpg_farm_{}.h5f'.format(i)))
            # Test the agent
            hist = agent.test(
                env,
                nb_episodes=10,
                action_repetition=1,
                nb_max_episode_steps=env.STEPS_PER_EPISODE,
                visualize=False
            )
            counter = counter + 1