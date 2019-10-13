from foolbox.distances import Distance
import numpy as np
import math

"""L2 Distance measure usable for foolbox attacks
"""

class L2_Distance(Distance):

    def _calculate(self):
        min_, max_ = self._bounds
        diff = (self.other - self.reference) / (max_ - min_)
        value = np.sqrt(np.sum(np.square(diff)))
        gradient = None
        return value, gradient

    @property
    def gradient(self):
        raise NotImplementedError