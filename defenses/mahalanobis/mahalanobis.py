import tensorflow as tf
import numpy as np
from defenses.mahalanobis.lib_generation import get_Mahalanobis_score_adv, sample_estimator
import torch
from torchvision import transforms
from utils.data_handling import get_data_loaders_cifar10, get_data_loaders_mnist
from torch.autograd import Variable
import os
import pickle

class MahalanobisDetector():

    """mahalanobis detector instance"""

    def __init__(self, threshold, model, dataset):
        self.threshold = threshold
        self.model = model
        self.dataset = dataset
        if dataset == "cifar10":
            self.regressor = pickle.load(open("defenses/mahalanobis/detector_values/lr_detector_cifar10_noise0,0005", 'rb'))
        if dataset == "mnist":
            self.regressor = pickle.load(open("defenses/mahalanobis/detector_values/lr_detector_mnist_noise0,0", 'rb'))
        self.sample_mean, self.precision = self.get_mean_and_precision(dataset)

    """apply defense
       images, labels: batch to compute score for
       returns scores of the logistic regressor trained for Mahalanobis score under mnist attack
    """
    def detect(self, images, labels):
        if (type(images) != torch.Tensor):
            images = torch.from_numpy(images)
        if (type(labels) != torch.Tensor):
            labels = torch.from_numpy(labels)
        scores = None
        for layer in range(5):
            layer_scores = np.array(get_Mahalanobis_score_adv(self.model, images, labels, 10, "resnet", self.sample_mean, self.precision, layer, 0.0005, dataset=self.dataset))
            if layer == 0:
                scores = layer_scores
            else:
                scores = np.column_stack([scores, layer_scores])
        reg_scores = self.regressor.predict_proba(np.reshape(scores, (len(images),5)))[:, 1]
        return reg_scores

    def get_mean_and_precision(self, dataset):
        '''
        mean and precision(inverse covariance) for all classes of data test set, needed to calculate mahalanobis score
        '''
        if(os.path.isfile("defenses/mahalanobis/detector_values/sample_mean_{}.npy".format(dataset))):
            sample_mean = np.load("defenses/mahalanobis/detector_values/sample_mean_{}.npy".format(dataset), allow_pickle=True)
            precision = np.load("defenses/mahalanobis/detector_values/precision_{}.npy".format(dataset), allow_pickle=True)

        else:
            # set information about feature extaction
            self.model.eval()
            if self.dataset == "mnist":
                temp_x = torch.rand(2, 1, 28, 28).cuda()
            else:
                temp_x = torch.rand(2, 3, 32, 32).cuda()
            temp_x = Variable(temp_x)
            temp_list = self.model.feature_list(temp_x)[1]
            num_output = len(temp_list)
            feature_list = np.empty(num_output)
            count = 0
            for out in temp_list:
                feature_list[count] = out.size(1)
                count += 1

            if self.dataset == "mnist":
                in_transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), ])
                train_loader, _ = get_data_loaders_mnist(batchsize=200, tf = in_transform)
            else:
                in_transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
                train_loader, _ = get_data_loaders_cifar10(batchsize=200, tf = in_transform)
            sample_mean, precision = sample_estimator(self.model, 10, feature_list, train_loader)
            np.save("defenses/mahalanobis/detector_values/sample_mean_{}.npy".format(dataset), sample_mean)
            np.save("defenses/mahalanobis/detector_values/precision_{}.npy".format(dataset), precision)

        return sample_mean, precision

