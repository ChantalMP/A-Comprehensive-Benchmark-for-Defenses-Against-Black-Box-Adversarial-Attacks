from utils.data_handling import get_data_loaders_cifar10, get_data_loaders_mnist
from utils.modelWrapper import ModelWrapper
import numpy as np
from classifiers.mymodel import MyModel
import os
from defenses.rce_kDensity.rce_kDensity import RCE_KDensity
from defenses.defensegan.defensegan_model import DefGanModel
import sys

'''
generator and getter for testsets for cifar10 and mnist used for evaluation
testset consists of random images classified correctly by given classifier
'''

def get_attack_testset(dataset):
    if os.path.isfile("attack/attack_testdata/{}_images_100.npy".format(dataset)) and os.path.isfile("attack/attack_testdata/{}_labels_100.npy".format(dataset)):
        testset = np.load("attack/attack_testdata/{}_images_100.npy".format(dataset))
        testlabels = np.load("attack/attack_testdata/{}_labels_100.npy".format(dataset))
        return testset, testlabels

#def generate_new_attack_testset(model_clean, model_mahalanobis, model_mahalanobis_noDefense, model_defgan, model_defgan_NoDefense, model_ens, model_ens_noDefense, model_rce, model_rce_NoDefense, model_ce, dataset, size = 10):
def generate_new_attack_testset(model_mahalanobis, model_mahalanobis_noDefense, model_rce, model_rce_NoDefense, model_ce, dataset, size=100):

    found_labels = []
    testset = None
    testlabels = []
    if dataset == "cifar10":
        _, testloader = get_data_loaders_cifar10(batchsize=1)
    elif dataset == "mnist":
        _, testloader = get_data_loaders_mnist(batchsize=1)

    testiter = iter(testloader)
    #f = open("usedLabels.txt", "w+")
    found_labels = [0 for i in range(10)]

    i = 0
    idx = -1
    while i < size:
        idx+=1
        data = next(testiter)
        inputs, labels = data
        image = inputs[0:1]
        label = labels[0]

        img = image.cpu().numpy()
        if found_labels[label] >= 10:
            continue
        #prediction_clean = model_clean.batch_predictions(img)

        prediction_m = model_mahalanobis.batch_predictions(img)
        prediction_m_noDef = model_mahalanobis_noDefense.batch_predictions(img)

        #prediction_defgan = model_defgan.batch_predictions(img)
        #prediction_defgan_noDef = model_defgan_NoDefense.batch_predictions(img)

        #prediction_ens = model_ens.batch_predictions(img)
        #prediction_ens_noDef = model_ens_noDefense.batch_predictions(img)

        prediction_rce = model_rce.batch_predictions(img)
        prediction_rce_noDef = model_rce_NoDefense.batch_predictions(img)
        prediction_ce = model_ce.batch_predictions(img)

        #if np.argmax(prediction_m).item() == label.item()\
         #       and np.argmax(prediction_m_noDef).item() == label.item()\
          #      and np.argmax(prediction_defgan).item() == label.item()\
           #     and np.argmax(prediction_defgan_noDef).item() == label.item() \
            #    and np.argmax(prediction_ens).item() == label.item() \
             #   and np.argmax(prediction_ens_noDef).item() == label.item() \
              #  and np.argmax(prediction_rce).item() == label.item() \
               # and np.argmax(prediction_rce_noDef).item() == label.item()\
                #and np.argmax(prediction_ce).item() == label.item():
        if np.argmax(prediction_m).item() == label.item()\
                and np.argmax(prediction_m_noDef).item() == label.item()\
                and np.argmax(prediction_rce).item() == label.item() \
                and np.argmax(prediction_rce_noDef).item() == label.item()\
                and np.argmax(prediction_ce).item() == label.item():

            found_labels[label] += 1
            #f.write("{} \n".format(idx))
            if testset is None:
                testset = np.array([image[0].numpy()])
            else:
                testset = np.concatenate([testset, np.array([image[0].numpy()])])
            testlabels.append(label.item())
            i+=1

    testlabels = np.array(testlabels)
    np.save("attack/attack_testdata/{}_images_100".format(dataset), testset)
    np.save("attack/attack_testdata/{}_labels_100".format(dataset), testlabels)

    #f.close()
    return testset, testlabels

if __name__ == '__main__':
    dataset = "cifar10"
    #images, labels = get_attack_testset(dataset)
    if dataset == "cifar10":
        classifier_path = "classifiers/cifar10_classifier/models/resnet44_200epochs_cifar10.pt"
        classifier_name = "resnet44_200epochs_cifar10.pt"
        k_threshold = -38
    elif dataset == "mnist":
        classifier_path = "classifiers/mnist_classifier/models/resnet44_2epochs_mnist.pt"
        classifier_name = "resnet44_2epochs_mnist.pt"
        classifier_path_m = "classifiers/mnist_classifier/models/state_epoch11_mahalanobis_mnist.pt"
        classifier_name_m = "state_epoch11_mahalanobis_mnist.pt"
        classifier_path_ens = ""
        classifier_name_ens = "modelA_ens"
        classifier_path_rce = ""
        classifier_name_rce = "rce"
        k_threshold = -105
    #classifier, _, _, _, _ = MyModel(dataset).get_model(model_name=classifier_name, model_path=classifier_path)
    classifier_m, _, _, _, _ = MyModel(dataset).get_model(model_name="mahalanobis_cifar10", model_path="")
    #classifier_m, _, _, _, _ = MyModel(dataset).get_model(model_name=classifier_name_m, model_path=classifier_path_m)
    #classifier_ens, _, _, _, _ = MyModel(dataset).get_model(model_name=classifier_name_ens, model_path=classifier_path_ens, model_type="Keras")
    #classifier_ens_noDefense, _, _, _, _ = MyModel(dataset).get_model(model_name="modelA", model_path="", model_type="Keras")
    classifier_rce = RCE_KDensity("rce", threshold = k_threshold, dataset=dataset)
    classifier_ce = RCE_KDensity("ce", threshold = k_threshold, dataset=dataset)
    #classifier_defgan = DefGanModel()

    #model_clean = ModelWrapper("PyTorch", classifier, dataset=dataset)
    model_mahalanobis = ModelWrapper("PyTorch", classifier_m, detector=["mahalanobis", 0.5], dataset=dataset, refset_name="mahalanobis")
    model_mahalanobis_noDefense = ModelWrapper("PyTorch", classifier_m, dataset=dataset, refset_name="mahalanobis")

    #model_defgan = ModelWrapper("DefGan", classifier_defgan, runTimeDefense="defgan", dataset=dataset, refset_name="defgan")
    #model_defgan_NoDefense = ModelWrapper("DefGan", classifier_defgan, dataset=dataset, refset_name="defgan")

    #model_ens = ModelWrapper("Keras", classifier_ens, dataset=dataset, refset_name="ensTraining")
    #model_ens_noDefense = ModelWrapper("Keras", classifier_ens_noDefense, dataset=dataset, refset_name="ensTraining")

    model_ce = ModelWrapper("RCEClassifier", classifier_ce, dataset=dataset, refset_name="rce")
    model_rce = ModelWrapper("RCEClassifier", classifier_rce, dataset=dataset, detector=["kdensity", k_threshold], refset_name="rce")
    model_rce_NoDefense = ModelWrapper("RCEClassifier", classifier_rce, dataset=dataset, refset_name="rce")

    #generate_new_attack_testset(model_mahalanobis, model_mahalanobis_noDefense, model_defgan, model_defgan_NoDefense, model_ens, model_ens_noDefense, model_rce, model_rce_NoDefense, model_ce, dataset)
    generate_new_attack_testset(model_mahalanobis, model_mahalanobis_noDefense, model_rce, model_rce_NoDefense, model_ce, dataset)