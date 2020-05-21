import numpy as np
from gym.spaces import Box

from ..core import ReferenceGenerator
from ..utils import instantiate


class SwitchedReferenceGenerator(ReferenceGenerator):
    """
    Reference Generator that switches randomly between multiple sub generators with a certain probability p for each.
    """

    reference_space = Box(-1, 1, shape=(1,))
    _reference = None
    _k = 0

    def __init__(self, sub_generators, sub_args=None, p=None, super_episode_length=(100, 10000), **kwargs):
        """
        Args:
            sub_generators(list(str/class/object)): List of keys, classes or objects to instantiate the sub_generators
            sub_args(dict/list(dict)/None): (Optional) Arguments to pass to the sub_converters. If not passed all kwargs
                will be passed to each sub_generator.
            p(list(float)/None): (Optional) Probabilities for each sub_generator. If None a uniform
                probability for each sub_converter is used.
            super_episode_length(Tuple(int, int): Minimum and maximum number of time steps a sub_generator is used.
            kwargs: All kwargs of the environment. Passed to the sub_generators, if no sub_args are passed.
        """
        super().__init__()
        if type(sub_args) is dict:
            sub_arguments = [sub_args] * len(sub_generators)
        elif hasattr(sub_args, '__iter__'):
            assert len(sub_args) == len(sub_generators)
            sub_arguments = sub_args
        else:
            sub_arguments = [kwargs] * len(sub_generators)
        self._sub_generators = [instantiate(ReferenceGenerator, sub_generator, **sub_arg)
                                for sub_generator, sub_arg in zip(sub_generators, sub_arguments)]
        self._probabilities = p or [1/len(sub_generators)] * len(sub_generators)
        self._current_episode_length = 0
        if type(super_episode_length) in [float, int]:
            super_episode_length = super_episode_length, super_episode_length + 1
        self._super_episode_length = super_episode_length
        self._current_ref_generator = self._sub_generators[0]

    def set_modules(self, physical_system):
        """
        Args:
            physical_system(PhysicalSystem): The physical system of the environment.
        """
        super().set_modules(physical_system)
        for sub_generator in self._sub_generators:
            sub_generator.set_modules(physical_system)
        ref_space_low = np.min([sub_generator.reference_space.low for sub_generator in self._sub_generators], axis=0)
        ref_space_high = np.max([sub_generator.reference_space.high for sub_generator in self._sub_generators], axis=0)
        self.reference_space = Box(ref_space_low, ref_space_high)
        self._referenced_states = self._sub_generators[0].referenced_states
        for sub_generator in self._sub_generators:
            assert np.all(sub_generator.referenced_states == self._referenced_states), \
                'Reference Generators reference different state variables'
            assert sub_generator.reference_space.shape == self.reference_space.shape, \
                'Reference Generators have differently shaped reference spaces'

    def reset(self, initial_state=None, initial_reference=None):
        self._reset_reference()
        return self._current_ref_generator.reset(initial_state, initial_reference)

    def get_reference(self, state, **kwargs):
        self._reference = self._current_ref_generator.get_reference(state, **kwargs)
        return self._reference

    def get_reference_observation(self, state, *_, **kwargs):
        if self._k >= self._current_episode_length:
            self._reset_reference()
            _, obs, _ = self._current_ref_generator.reset(state, self._reference)
        else:
            obs = self._current_ref_generator.get_reference_observation(state, **kwargs)
        self._k += 1
        return obs

    def _reset_reference(self):
        self._current_episode_length = np.random.randint(self._super_episode_length[0], self._super_episode_length[1])
        self._k = 0
        self._current_ref_generator = np.random.choice(self._sub_generators, p=self._probabilities)
