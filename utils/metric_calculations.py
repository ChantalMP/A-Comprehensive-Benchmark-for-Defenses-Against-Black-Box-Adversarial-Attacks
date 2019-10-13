from utils.modelWrapper import ModelWrapper
from utils.data_handling import get_data_loaders_mnist, get_data_loaders_cifar10
import torchvision.transforms as transforms
import os
from defenses.defensegan.defensegan_model import DefGanModel
import numpy as np
from classifiers.mymodel import MyModel
import torch
from defenses.rce_kDensity.rce_kDensity import RCE_KDensity
import matplotlib.pyplot as plt

def calc_accuracy(model:ModelWrapper, ref_name, dataset = "mnist", tf=transforms.ToTensor(), detect=None):
    if dataset == "mnist":
        _, testloader = get_data_loaders_mnist(128, tf=tf)
    else:
        _, testloader = get_data_loaders_cifar10(128, tf=tf)
    #adv_img = torch.from_numpy(np.load("defenses/rce_kDensity/data/Adv_mnist_fgsm.npy")).transpose(3, 1)
    # for i in range(7):
    #     plt.imshow(np.squeeze(adv_img[i].numpy()))
    #     plt.savefig("testi{}.png".format(i))
    if not os.path.isdir("evaluation_results/accuracies"):
        os.mkdir("evaluation_results/accuracies")
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        if(type(labels) == torch.Tensor):
            labels = labels.numpy()
            images = images.numpy()
        pred = model.batch_predictions(images, detect = detect)#adv_img[:128].numpy()
        predicted = np.argmax(pred, axis=1)
        total += len(labels)
        correct += (predicted == labels).sum()

    file = open("evaluation_results/accuracies/{}".format(ref_name), "w")
    file.write('Accuracy of the network on the 10000 test images: {} \n'.format(
        100 * correct * 1.0 / total * 1.0))
    print('Accuracy of the network on the 10000 test images: {} \n'.format(
        100.0 * correct.item() * 1.0 / total * 1.0))
    print("results saved to: {}".format("evaluation_results/accuracies/{}".format(ref_name)))

if __name__ == '__main__':
    dataset = 'mnist'
    if dataset == "cifar10":
        classifier_path = "classifiers/cifar10_classifier/models/resnet44_200epochs_cifar10.pt"
        classifier_name = "resnet44_200epochs_cifar10.pt"
    elif dataset == "mnist":
        classifier_path = "classifiers/mnist_classifier/models/resnet44_2epochs_mnist.pt"
        classifier_name = "resnet44_2epochs_mnist.pt"
        classifier_path_m = "classifiers/mnist_classifier/models/state_epoch11_mahalanobis_mnist.pt"
        classifier_name_m = "state_epoch11_mahalanobis_mnist.pt"
        classifier_path_ens = ""
        classifier_name_ens = "modelA_ens"
        classifier_name_ens_noDefense = "modelA"
        classifier_path_rce = ""
        classifier_name_rce = "rce"
    #classifier, _, _, _, _ = MyModel(dataset).get_model(model_name=classifier_name, model_path=classifier_path)
    #classifier_m, _, _, _, _ = MyModel(dataset).get_model(model_name=classifier_name_m, model_path=classifier_path_m)
    #classifier_m_cifar10, _, _, _, _ = MyModel(dataset).get_model(model_name="mahalanobis_cifar10", model_path="")

    #classifier_ens, _, _, _, _ = MyModel(dataset).get_model(model_name=classifier_name_ens,model_path=classifier_path_ens, model_type="Keras")
    #classifier_ens_noDefense, _, _, _, _ = MyModel(dataset).get_model(model_name=classifier_name_ens_noDefense,model_path=classifier_path_ens, model_type="Keras")
    classifier_defgan = DefGanModel()
    #classifier_rce = RCE_KDensity("rce", -105, dataset=dataset)
    #classifier_ce_cifar10 = RCE_KDensity("ce", 0, dataset="cifar10")
    #classifier_rce_cifar10 = RCE_KDensity("rce", 0, dataset=dataset)


    #model_clean = ModelWrapper("PyTorch", classifier, dataset=dataset)
    #model_mahalanobis = ModelWrapper("PyTorch", classifier_m, detector=["mahalanobis", 0.5], dataset=dataset, refset_name="mahalanobis")
    #model_mahalanobis_noDefense = ModelWrapper("PyTorch", classifier_m, dataset=dataset, refset_name="mahalanobis")

    #model_mahalanobis_cifar10 = ModelWrapper("PyTorch", classifier_m_cifar10, detector=["mahalanobis", 0.5], dataset=dataset, refset_name="mahalanobis")
    #model_mahalanobis_cifar10_noDefense = ModelWrapper("PyTorch", classifier_m_cifar10, dataset=dataset, refset_name="mahalanobis")

    model_defgan = ModelWrapper("DefGan", classifier_defgan, runTimeDefense="defgan", dataset=dataset)
    #model_defgan_noDefense = ModelWrapper("DefGan", classifier_defgan, dataset=dataset)

    #model_ens = ModelWrapper("Keras", classifier_ens, dataset=dataset)
    #model_ens_noDefense = ModelWrapper("Keras", classifier_ens_noDefense, dataset=dataset)

    #model_rce = ModelWrapper("RCEClassifier", classifier_rce, dataset=dataset, detector=["kdensity",-105])
    #model_ce_cifar10 = ModelWrapper("RCEClassifier", classifier_ce_cifar10, dataset=dataset)
    #model_rce_cifar10 = ModelWrapper("RCEClassifier", classifier_rce_cifar10, dataset=dataset)
    #model_rce_cifar10_kdensity = ModelWrapper("RCEClassifier", classifier_rce_cifar10, dataset=dataset, detector=["kdensity", -38])



    #calc_accuracy(model_clean, "model_clean")
    #calc_accuracy(model_mahalanobis, "model_mahalanobis_with_defense", detect=True)
    #calc_accuracy(model_mahalanobis_noDefense, "model_mahalanobis_with_defense")
    #calc_accuracy(model_ens, "model_ens_with_defense")
    #calc_accuracy(model_defgan_noDefense, "model_defgan_noDefense")
    calc_accuracy(model_defgan, "model_defgan")
    #calc_accuracy(model_mahalanobis_cifar10_noDefense, "model_mahalanobis_cifar10_noDefense", dataset=dataset, detect=False)
    #calc_accuracy(model_mahalanobis_cifar10, "model_mahalanobis_cifar10", dataset=dataset, detect=True)
    #calc_accuracy(model_ce_cifar10, "model_ce_cifar10", dataset=dataset)
    #calc_accuracy(model_rce_cifar10, "model_rce_cifar10", dataset=dataset)
    #calc_accuracy(model_rce_cifar10_kdensity, "model_rce_cifar10_kdensity", dataset=dataset, detect=True)

    #calc_accuracy(model_rce, "model_rce_mnist_kdensity", dataset=dataset, detect=True)


