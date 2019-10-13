import tensorflow as tf
import defenses.rce_kDensity.resnet_model_mnist as resnet_model_mnist
import defenses.rce_kDensity.resnet_model_cifar as resnet_model_cifar
from defenses.rce_kDensity.kdensity.detect.util import get_deep_representations, score_samples
from sklearn.neighbors import KernelDensity
from utils.data_handling import get_data_loaders_mnist, get_data_loaders_cifar10
import numpy as np
import math
import pickle
import os

class RCE_KDensity():

    """Used for prediction of network trained with RCE and K-Density detection instance"""

    def __init__(self, classifier_name, threshold = None, dataset = 'mnist'):
        print("OLD!!!!!!!!!!!!!!!!")
        self.threshold = threshold
        self.dataset = dataset
        self.classifier_name = classifier_name
        self.kdes = self.train_detector()


    """builds the computation graph and runs it to get tensorflow predictions
    images has to have shape: (batchsize, len, width, channels)"""
    def predict_rce(self, images):
        rce = (self.classifier_name == "rce")
        self.hps = resnet_model_mnist.HParams(batch_size=len(images),
                                              num_classes=10,
                                              min_lrn_rate=0.0001,
                                              lrn_rate=0.1,
                                              num_residual_units=5,
                                              use_bottleneck=False,
                                              weight_decay_rate=0.000,
                                              relu_leakiness=0.1,
                                              optimizer='mom',
                                              RCE_train=rce)
        tf.reset_default_graph()
        if self.dataset == "mnist":
            images = images - 0.5
            model = resnet_model_mnist.ResNet(self.hps, images, 'eval', Reuse=False)
            model_name = "defenses/rce_kDensity/models_{}/resnet32_{}/model.ckpt-20001".format(
            self.dataset, self.classifier_name)
        else:
            images = images - 0.5
            model = resnet_model_cifar.ResNet(self.hps, images, 'eval', Reuse=False)
            model_name = "defenses/rce_kDensity/models_{}/resnet32_{}/model.ckpt-90001".format(
            self.dataset, self.classifier_name)
        model.build_graph()
        saver = tf.train.Saver()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        tf.train.start_queue_runners(self.sess)
        saver.restore(self.sess, model_name)
        predictions, last_hidden_features = self.sess.run([model.predictions, model.t_SNE_logits])
        return predictions, last_hidden_features

    def train_detector(self):
        if (os.path.isfile("defenses/rce_kDensity/kdensity/{}/kDensityEstimatorFull".format(self.dataset))):
            kdes = pickle.load(open("defenses/rce_kDensity/kdensity/{}/kDensityEstimatorFull".format(self.dataset), 'rb'))
            return kdes
        ## Get KDE scores
        # Get deep feature representations
        print('Getting deep feature representations...')
        # TODO how to handle batches, can't get whole test set here
        if self.dataset == "mnist":
            trainloader, _ = get_data_loaders_mnist()
        else:
            trainloader, _ = get_data_loaders_cifar10()
        X_train_features = None
        for data in trainloader:
            X_train, Y = data
            if X_train_features is None:
                _, X_train_features = self.predict_rce(np.transpose(X_train, (0,2,3,1)))
                Y_train = Y.numpy()
            else:
                _, new_features = self.predict_rce(np.transpose(X_train, (0, 2, 3, 1)))
                X_train_features = np.concatenate((X_train_features, new_features))
                Y_train = np.concatenate((Y_train, Y.numpy()))

        # Train one KDE per class
        print('Training KDEs...')
        #to one hot encoding
        Y_train = np.eye(10)[Y_train]
        class_inds = {}
        for i in range(Y_train.shape[1]):
            class_inds[i] = np.where(Y_train.argmax(axis=1) == i)[0]
        kdes = {}
        bandwidth = math.sqrt(0.1 / 0.26)  # from paper
        for i in range(Y_train.shape[1]):
            kdes[i] = KernelDensity(kernel='gaussian',
                                    bandwidth=bandwidth) \
                .fit(X_train_features[class_inds[i]])

        pickle.dump(kdes, open("defenses/rce_kDensity/kdensity/{}/kDensityEstimatorFull".format(self.dataset), 'wb'))
        return kdes

    def detect(self, features, labels):

        # Get density estimates
        #print('computing densities...')
        labels = np.argmax(labels, axis=1)
        scores = score_samples(
            self.kdes,
            features,
            labels
        )

        return scores