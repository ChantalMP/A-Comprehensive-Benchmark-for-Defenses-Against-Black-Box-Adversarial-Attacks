'''generates mnist adversarial FGSM samples against the rce trained model'''

from foolbox.attacks import FGSM
from foolbox.criteria import Misclassification
from defenses.rce_kDensity.rce_kDensity_old import RCE_KDensity
from utils.modelWrapper import ModelWrapper
from utils.data_handling import get_data_loaders_mnist
import numpy as np

if __name__ == '__main__':
    classifier_rce = RCE_KDensity()
    model = ModelWrapper("Tensorflow", classifier_rce, dataset='mnist')
    criterion = Misclassification()
    attack = FGSM(model, criterion=criterion)
    trainloader, testloader = get_data_loaders_mnist(128)

    i = 0
    adversarials = None
    for data in testloader:
        images, labels = data
        for image, label in zip(images, labels):
            adversarial = attack(image.numpy(), label=label.numpy())
            if adversarials is None:
                adversarials = adversarial
            else:
                adversarials = np.append(adversarials, adversarial)