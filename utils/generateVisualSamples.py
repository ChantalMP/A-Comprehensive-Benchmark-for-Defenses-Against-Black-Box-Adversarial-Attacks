import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator
from utils.image_utils import plot_image

def saveSVG(imagepath, adversarialpath, name):
    plt.gray()
    image = np.load(imagepath)
    adversarial = np.load(adversarialpath)

    plt.subplot(3, 3, 1)
    image = np.transpose(image[9], (1, 2, 0))
    plt.imshow(np.squeeze(image))
    plt.axis('off')

    plt.subplot(3, 3, 2)
    adversarial = np.transpose(adversarial[9], (1, 2, 0))
    plt.imshow(np.squeeze(adversarial))
    plt.axis('off')

    plt.subplot(3, 3, 3)
    difference = np.squeeze(adversarial - image)
    plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
    plt.axis('off')

    plt.savefig("samples/{}.png".format(name), bbox_inches='tight', pad_inches=0)

def plot_multiple_samples(adversarialpath, name):
    plt.gray()
    adversarial = np.load(adversarialpath)

    plt.subplot(3, 3, 1)
    image1 = np.transpose(adversarial[27], (1, 2, 0))
    plt.imshow(np.squeeze(image1))
    plt.axis('off')

    plt.subplot(3, 3, 2)
    image2 = np.transpose(adversarial[32], (1, 2, 0))
    plt.imshow(np.squeeze(image2))
    plt.axis('off')

    plt.subplot(3, 3, 3)
    image3 = np.transpose(adversarial[44], (1, 2, 0))
    plt.imshow(np.squeeze(image3))
    plt.axis('off')

    plt.savefig("samples/{}.png".format(name), bbox_inches='tight', pad_inches=0)

def print_labels():
    import os
    label_file = open("labels_new.txt", "w+")
    for f in os.listdir("evaluation_results/data"):
        if "labels" in f:
            labels = np.load(os.path.join("evaluation_results/data",f))
            label_file.write("{}: {}\n".format(f, labels))



if __name__ == '__main__':
    #saveSVG("evaluation_results/data/orig_image_cifar10_bapp_100000_MSE_RCE_KDensity_100.npy",
    #        "evaluation_results/data/adv_images_cifar10_bapp_100000_MSE_RCE_KDensity_100.npy",
    #        "cifar10_rce_rceAndK")
    #plot_multiple_samples("evaluation_results/data/adv_images_mnist_bapp_100000_MSE_EnsAdvTraining_100_newattack.npy", "adv_samples_enstrain")
    plot_multiple_samples("evaluation_results/data/orig_image_mnist_bapp_100000_MSE_EnsAdvTraining_100_newattack.npy","orig_samples_enstrain")

    #print_labels()

    images_orig = np.load("evaluation_results/data/orig_image_mnist_bapp_100000_MSE_EnsAdvTraining_100.npy")
    images_adv = np.load("evaluation_results/data/adv_images_mnist_bapp_100000_MSE_EnsAdvTraining_100.npy")
    labels = np.load("evaluation_results/data/adv_labels_mnist_bapp_100000_MSE_EnsAdvTraining_100.npy")
    labels_orig = np.load("evaluation_results/data/orig_labels_mnist_bapp_100000_MSE_EnsAdvTraining_100.npy")
    a = 1
