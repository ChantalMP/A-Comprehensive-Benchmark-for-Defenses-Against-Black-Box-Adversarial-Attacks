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
        self.threshold = threshold
        self.dataset = dataset
        self.classifier_name = classifier_name
        self.kdes = self.train_detector()

        self.model_per_batch = {}
        self.model_name = ''
        self.sess_per_batch = {}


    """builds the computation graph and runs it to get tensorflow predictions
    images has to have shape: (batchsize, len, width, channels)"""
    def predict_rce(self, images):
        rce = (self.classifier_name == "rce")
        #for cifar10 HParams are the same
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

        batch_size = images.shape[0]
        images = images - 0.5

        # We should build a model if it doesn't exist, also create a matching sessions
        if batch_size not in self.model_per_batch:
            tf.reset_default_graph()
            #shape = images.shape
            images_pl = tf.placeholder(tf.float32, shape=(batch_size,) + images.shape[1:], name='images_pl')
            if self.dataset == "mnist":

                self.model_per_batch[batch_size] = resnet_model_mnist.ResNet(self.hps, images_pl, 'eval', Reuse=False)
                self.model_name = "defenses/rce_kDensity/models_{}/resnet32_{}/model.ckpt-20001".format(
                    self.dataset, self.classifier_name)
            else:
                self.model_per_batch[batch_size] = resnet_model_cifar.ResNet(self.hps, images_pl, 'eval', Reuse=False)
                self.model_name = "defenses/rce_kDensity/models_{}/resnet32_{}/model.ckpt-90001".format(
                    self.dataset, self.classifier_name)
            self.model_per_batch[batch_size].build_graph()

            saver = tf.train.Saver()
            self.sess_per_batch[batch_size] = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            tf.train.start_queue_runners(self.sess_per_batch[batch_size])
            saver.restore(self.sess_per_batch[batch_size], self.model_name)

        predictions, last_hidden_features = self.sess_per_batch[batch_size].run([self.model_per_batch[batch_size].predictions, self.model_per_batch[batch_size].t_SNE_logits],feed_dict={'images_pl:0':images})

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