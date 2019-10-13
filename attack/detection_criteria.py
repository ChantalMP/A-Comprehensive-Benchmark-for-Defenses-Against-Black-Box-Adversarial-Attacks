import numpy as np
import math
from foolbox.criteria import Criterion, CombinedCriteria

"""L2 Distance measure usable for foolbox attacks
"""

class Detection_criteria(Criterion):

    def name(self):
        return 'DetectionCriteria'

    def is_adversarial(self, predictions, label):
        top1 = np.argmax(predictions)
        return top1 != label and top1 != 10