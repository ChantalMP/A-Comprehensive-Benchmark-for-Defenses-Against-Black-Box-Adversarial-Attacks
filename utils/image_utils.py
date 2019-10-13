import matplotlib
matplotlib.use('Agg') #https://stackoverflow.com/a/37605654
import matplotlib.pyplot as plt

import numpy as np
from utils.data_handling import classes_cifar10, classes_mnist, get_data_loaders_mnist, get_data_loaders_cifar10
import os
from typing import List, Tuple
from classifiers.mymodel import MyModel
from utils.modelWrapper import ModelWrapper
from defenses.defensegan.defensegan_model import DefGanModel
from defenses.rce_kDensity.rce_kDensity import RCE_KDensity

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

"""
plots the given images, their adverarial counterparts and the difference between and saves this to file with given filename
if filename == "" just shows plot without saving
"""
def plot_result(orig_labels:List, adv_labels:List, images, adversarials:List, filename:str =""):
    #classes = classes_mnist
    classes = classes_cifar10
    plt.figure()
    row = 0
    for i in range(len(images)):
        plt.subplot(3, 3, row+1)
        label = orig_labels[i]
        plt.title('Orig: {} - {}'.format(label, classes[label]))
        image = np.transpose(images[i], (1,2,0))
        plt.imshow(np.squeeze(image))
        plt.axis('off')

        plt.subplot(3, 3, row+2)
        adv_label = adv_labels[i]
        plt.title('Adv: {} - {}'.format(adv_label, classes[adv_label]))
        adversarial = np.transpose(adversarials[i], (1,2,0))
        plt.imshow(np.squeeze(adversarial))
        plt.axis('off')

        plt.subplot(3, 3, row+3)
        plt.title('Diff')
        difference = np.squeeze(adversarial - image)
        plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
        plt.axis('off')
        row += 3

    if filename == "":
        plt.show()
    else:
        plt.savefig("evaluation_results/"+filename)


"""
function for finding most similar image belonging to other class from a subset of the testset in terms of L2 norm
goal_img: image you want to perturb
img_label: original label of goal_img
reference_images: subset of test images where to search for closest image
reference_labels: labels for reference_images
"""

def find_closest_adversarial_reference_image(goal_img:np.array, reference_images:np.array, reference_labels:np.array, model) -> np.array:
    img_label = np.argmax(model.batch_predictions(np.expand_dims(goal_img, axis=0), detect=False, foolbox=False))
    maskout_correct_labels = reference_labels != img_label
    ref_images_masked = reference_images[maskout_correct_labels]
    difference_metric = np.linalg.norm(np.reshape(ref_images_masked - goal_img, (ref_images_masked.shape[0], -1)), axis=1)
    most_similar_index = np.argmin(difference_metric)
    most_similar_image = ref_images_masked[most_similar_index]
    most_similar_pred = np.argmax(model.batch_predictions(np.expand_dims(most_similar_image, axis=0), detect=True, foolbox=False))
    while most_similar_pred == img_label or most_similar_pred == 10:
        difference_metric[most_similar_index] = np.max(difference_metric)
        most_similar_index = np.argmin(difference_metric)
        most_similar_image = ref_images_masked[most_similar_index]
        most_similar_pred = np.argmax(model.batch_predictions(np.expand_dims(most_similar_image, axis=0), detect=True, foolbox=False))

    return most_similar_image


"""
function for getting a subset of the test images
"""
def get_reference_set(dataset:str, refset_name:str = "none") -> Tuple[np.array, np.array]:
    if os.path.isfile("attack/attack_testdata/{}_refset_{}_images.npy".format(dataset, refset_name)) and os.path.isfile("attack/attack_testdata/{}_refset_{}_labels.npy".format(dataset, refset_name)):
        testset = np.load("attack/attack_testdata/{}_refset_{}_images.npy".format(dataset, refset_name))
        testlabels = np.load("attack/attack_testdata/{}_refset_{}_labels.npy".format(dataset, refset_name))
    else:
        print("new refset")
        if dataset == "cifar10":
            _, testloader = get_data_loaders_cifar10(batchsize=2000)
        elif dataset == "mnist":
            _, testloader = get_data_loaders_mnist(batchsize=2000)
        data = next(iter(testloader))
        inputs, labels = data
        testset = inputs.numpy()
        testlabels = labels.numpy()
        np.save("attack/attack_testdata/{}_referenceset_pics".format(dataset), testset)
        np.save("attack/attack_testdata/{}_referenceset_labels".format(dataset), testlabels)

    return testset, testlabels

#import resource
#rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

'''
for generating a reference set of the testset
'''

def generate_refSet(dataset = "mnist", defense = "none"):

    found_labels = []
    testset = None
    testlabels = []
    if dataset == "cifar10":
        _, testloader = get_data_loaders_cifar10(batchsize=1)
    elif dataset == "mnist":
        _, testloader = get_data_loaders_mnist(batchsize=1)

    testiter = iter(testloader)
    i = 0

    if defense == "ensTraining":
        classifier_ens, _, _, _, _ = MyModel(dataset).get_model(model_name="modelA_ens", model_path="", model_type="Keras")
        model = ModelWrapper("Keras", classifier_ens, dataset=dataset)
        classifier_ens_noDefense, _, _, _, _ = MyModel(dataset).get_model(model_name="modelA", model_path="", model_type="Keras")
        model_noDefense = ModelWrapper("Keras", classifier_ens_noDefense, dataset=dataset)

    elif defense == "defgan":
        classifier_defgan = DefGanModel()
        model = ModelWrapper("DefGan", classifier_defgan, runTimeDefense="defgan", dataset=dataset)
        model_noDefense = ModelWrapper("DefGan", classifier_defgan, dataset=dataset)

    elif defense == "mahalanobis":
        classifier_path_m = "classifiers/mnist_classifier/models/state_epoch11_mahalanobis_{}.pt".format(dataset)
        classifier_name_m = "state_epoch11_mahalanobis_mnist.pt" if dataset == "mnist" else "mahalanobis_cifar10"
        classifier_m, _, _, _, _ = MyModel(dataset).get_model(model_name=classifier_name_m, model_path=classifier_path_m)
        model = ModelWrapper("PyTorch", classifier_m, detector=["mahalanobis", 0.5], dataset=dataset, refset_name="mahalanobis")
        model_noDefense = ModelWrapper("PyTorch", classifier_m, dataset=dataset, refset_name="mahalanobis")

    elif defense == "rce":
        classifier_rce = RCE_KDensity("rce", threshold=-38, dataset=dataset)
        classifier_ce = RCE_KDensity("ce", threshold=-38, dataset=dataset)
        model_noDefense = ModelWrapper("RCEClassifier", classifier_ce, dataset=dataset)
        model_only_rce = ModelWrapper("RCEClassifier", classifier_rce, dataset=dataset, detector=["kdensity", -38])
        model = ModelWrapper("RCEClassifier", classifier_rce, dataset=dataset)

    count = 0
    while i < 2000:
        if (i %100 == 0):
            print(count)
        count+=1
        try:
            data = next(testiter)
        except Exception as e:
            print(count)
            print(i)
        inputs, labels = data
        image = inputs[0:1]
        label = labels[0]

        prediction = model.batch_predictions(image.cpu().numpy())
        prediction_noDefense = model_noDefense.batch_predictions(image.cpu().numpy())
        only_rce = True
        if (defense == "rce"):
            prediction_only_rce = model_only_rce.batch_predictions(image.cpu().numpy())
            only_rce = (np.argmax(prediction_only_rce).item() == label.item())
        if np.argmax(prediction).item() == label.item()\
                and np.argmax(prediction_noDefense).item() == label.item()\
                and only_rce:
            found_labels.append(label)
            if testset is None:
                testset = np.array([image[0].numpy()])
            else:
                testset = np.concatenate([testset, np.array([image[0].numpy()])])
            testlabels.append(label.item())
            i+=1

    testlabels = np.array(testlabels)
    np.save("attack/attack_testdata/{}_refset_{}_images".format(dataset, defense), testset)
    np.save("attack/attack_testdata/{}_refset_{}_labels".format(dataset, defense), testlabels)

def plot_image(image, name):
    image = np.transpose(image, (1, 2, 0))
    plt.imshow(np.squeeze(image))
    plt.savefig("debug/{}".format(name))

if __name__ == '__main__':
    #generate reference sets for all model pairs (with and without defense)
    #generate_defgan_refSet()
    generate_refSet("cifar10", "rce")
