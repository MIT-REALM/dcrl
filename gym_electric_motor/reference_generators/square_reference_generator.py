import numpy as np
from scipy import signal as sg
from .subepisoded_reference_generator import SubepisodedReferenceGenerator


class SquareReferenceGenerator(SubepisodedReferenceGenerator):
    """
    Reference Generator that generates a square waveform with deterministic parameters.
    """

    _amplitude = 0
    _frequency = 0

    def __init__(self, amplitude=None, frequency=2.5, *_, **kwargs):
        """
        Args:
            amplitude(float): The amplitude of square wave.
            frequency(float): The frequency of square wave.
            kwargs(dict): Arguments passed to the superclass SubepisodedReferenceGenerator .
        """
        super().__init__(**kwargs)
        self._amplitude = amplitude or np.inf
        self._frequency = frequency

    def set_modules(self, physical_system):
        super().set_modules(physical_system)
        # but amplitude and offset cannot exceed limit margin
        self._amplitude = np.clip(
            self._amplitude, 0, self._limit_margin[1] * 0.7 )

    def _sawtooth(self, t, width=1.0):
        state_ref = (1 + sg.sawtooth(2*np.pi * self._frequency * t, width)) / 2
        return state_ref

    def _sine(self, t):
        state_ref = (1 - np.cos(2*np.pi * self._frequency * t)) / 2
        return state_ref

    def _square(self, t):
        state_ref = sg.square(2*np.pi * self._frequency * t)
        return state_ref

    def _sawtooth_sine(self, t):
        state_ref = 0.75 * self._sawtooth(t, 0.7) + 0.2 * self._sine(4*t)
        return state_ref

    def _reset_reference(self):
        self._current_episode_length = self._episode_len_range[1]
        t = np.linspace(0, (self._current_episode_length - 1) * self._physical_system.tau, self._current_episode_length)
        self._reference = self._amplitude * self._sawtooth(t)
        self._reference = np.clip(self._reference, 0, self._limit_margin[1])
