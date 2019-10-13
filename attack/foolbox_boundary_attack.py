from foolbox.criteria import TargetClass, Misclassification
from foolbox.distances import Linfinity, MSE
from attack.L2_distance import L2_Distance
from attack.detection_criteria import Detection_criteria
import foolbox
import numpy as np
import math
from utils.image_utils import find_closest_adversarial_reference_image, get_reference_set

"""Class for executing the iterative attack using the foolbox framework
   Depending on given config file either executes BoundaryAttack or BoundaryAttack++
"""

#distnace string to distance
distances = {"Linfinity": Linfinity, "L2_Distance" : L2_Distance, "MSE" : MSE}
#thresholds aimed depending on used distance measures
distWithThreshold_cifar10 = {Linfinity: 0.05, L2_Distance : math.sqrt((0.05**2)*28*28), MSE : 0.0005}
distWithThreshold_mnist = {Linfinity: 0.05, L2_Distance : math.sqrt((0.05**2)*28*28), MSE : 0.0005}


'''
image: image you want to perturb
label: original image label
model: pyTorch classifier to attack
target_class: -1 for untargeted attack, class label for targeted attack
iterations: attack iterations
distance: used distance measure
'''
def run_attack(image, label, model, target_class, max_queries, distance, dataset, attack, refset_name="none", model_name = "none"):

    # get closest starting point picture
    data, labels = get_reference_set(dataset, refset_name=refset_name)
    init_pic = find_closest_adversarial_reference_image(image ,data, labels, model)

    # get the prediction of the model to that image as integer
    image = np.float32(image)
    adversarial = None
    criterion = Detection_criteria() if target_class == -1 else TargetClass(target_class)

    distWithThreshold = distWithThreshold_mnist if dataset == 'mnist' else distWithThreshold_cifar10
    with open("{}_log.csv".format(model_name), 'w+') as f:
        mse = MSE(image, init_pic, bounds=(0,1))
        f.write("-1,0,{}\n".format(mse._calculate()[0]))
    attack = attack(model, criterion=criterion, distance=distances[distance])#, threshold=distWithThreshold[distances[distance]])
    try:
        adversarial = attack(image, label=label, verbose=True, log_every_n_steps=10, unpack=False, batch_size = 64, starting_point=init_pic, max_queries=max_queries, log_name = "{}_log.csv".format(model_name), stay_at_best = True)
    except Exception as e:
        print(e)
        return None, 0
    done_iters = model.queries
    return adversarial, done_iters