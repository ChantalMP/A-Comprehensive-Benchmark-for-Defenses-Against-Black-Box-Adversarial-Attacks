from foolbox.models import Model
import torch
import os
import copy
from defenses.defensegan.models.gan import MnistDefenseGAN
from defenses.defensegan.utils.config import load_config
from defenses.defensegan.defensegan import DefenseGanDefense
import numpy as np
from defenses.mahalanobis.mahalanobis import MahalanobisDetector
from torchvision import transforms
from typing import Optional, Tuple
from defenses.rce_kDensity.rce_kDensity import RCE_KDensity
import tensorflow as tf

class ModelWrapper(Model):

    '''
    Foolbox model wrapper
    classifier_type: model type of classifier (pyTorch or tensorflow)
    classifier: classification model to use
    runTimeDefense: defense applied before images are given to classifier if any
    detector: detector applied after predicting, to decide if given image(with label) is adversarial and belonging threshold if scorebased: (detector, threshold)
    '''
    def __init__(self, classifier_type:str, classifier, runTimeDefense:Optional[str]=None, detector:Optional[str]=None, num_classes:int=11, channel_axis:int=1, bounds:Tuple[int,int] = (0,1), preprocessing:Tuple[int,int]=(0, 1), dataset:str="cifar10", refset_name=""):
        super().__init__(bounds, channel_axis, preprocessing)
        self.classifier = classifier
        if classifier_type == "PyTorch":
            self.classifier.eval()
        self.classifier_type = classifier_type
        self.num_classes = num_classes
        self.queries = 0
        self.runTimeDefense = None
        self.detector = None
        self.dataset = dataset
        self.refset_name = refset_name
        if (runTimeDefense == "defgan"):
            self.runTimeDefense = self.init_defensegan()
        if (detector != None):
            if (detector[0] == "mahalanobis"):
                self.detector = MahalanobisDetector(detector[1], self.classifier, dataset)
            elif (detector[0] == "kdensity"):
                self.detector = RCE_KDensity("rce", detector[1], dataset=dataset)

    def num_classes(self):
        return self.num_classes

    """
    prediction function
    imgs: batch of images to predict class after applying defense if any
    """
    def batch_predictions(self, imgs:np.array, detect:bool = True, foolbox = True) -> np.array:
        images = copy.deepcopy(imgs)
        if self.refset_name == "mahalanobis":
            #for this detectors models are trained with different normalisations, so we need to apply those in the forward pass as well
            if self.dataset == 'mnist':
                tf = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,)), ])
            elif self.dataset == 'cifar10':
                tf = transforms.Compose(
                    [transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
            if type(images) != torch.Tensor:
                images = torch.from_numpy(images)
            for i in range(len(images)):
                images[i] = tf(images[i])
            images = images.numpy()
        if foolbox:
            self.queries += len(imgs)
        if self.runTimeDefense is not None:
            images = self.runTimeDefense.defend(images)
        if self.classifier_type == "PyTorch":
            predictions =  self.get_pytorch_predictions(images)
        elif self.classifier_type == "Keras":
            predictions = self.get_keras_predictions(images)
        elif self.classifier_type == "RCEClassifier":
            predictions, last_hidden_features = self.get_tensorflow_rce_predictions(images)
        elif self.classifier_type =="DefGan":
            predictions = self.get_tensorflow_defgan_predictions(images)
        #row for label "adversarial"
        zeros = np.zeros((len(images), 1))
        zeros = zeros - np.abs(np.min(predictions)) -1
        predictions = np.append(predictions, zeros, axis=1)
        #compute detection scores
        if (detect == True and self.detector != None):
            if type(self.detector) == MahalanobisDetector:
                detection_scores = self.detector.detect(images, predictions) #TODO make normalisation here
                mask = [detection_score > self.detector.threshold for detection_score in detection_scores]
            elif type(self.detector) == RCE_KDensity:
                detection_scores = self.detector.detect(last_hidden_features, predictions)
                if self.dataset == "mnist":
                    mask = [detection_score < self.detector.threshold for detection_score in detection_scores]
                else:
                    mask = [detection_score > self.detector.threshold for detection_score in detection_scores]

            #set to label 11 in one hot encoding if prediction is greater 0.5 -> high probability to be adversarial - 11 stands for "adversarial"
            predictions[mask] = np.array([int(i == 10) for i in range(11)])
        return predictions#, last_hidden_features

    def get_keras_predictions(self, images):
        predictions = self.classifier.predict(np.transpose(images, (0,2,3,1)))
        return predictions

    def get_pytorch_predictions(self, images):
        if (type(images) != torch.Tensor):
            images = torch.from_numpy(images)
        predictions = self.classifier(images.cuda()).detach().cpu().numpy()
        return predictions

    def get_tensorflow_rce_predictions(self, images):
        #TODO think about this
        predictions, last_hidden_features = self.classifier.predict_rce(np.transpose(images, (0,2,3,1)))
        factor = -1 if self.classifier.classifier_name == "rce" else 1
        return predictions*factor, last_hidden_features

    def get_tensorflow_defgan_predictions(self, images):
        return self.classifier.predict(np.transpose(images,(0,2,3,1)))

    """
    initialisation of defensegan runtime defense
    """
    def init_defensegan(self):
        GAN = MnistDefenseGAN
        path = os.path.abspath("defenses/defensegan/checkpoints/mnist")
        cfg = load_config(path)
        gan = GAN(cfg=cfg, test_mode=True)
        gan.load_generator()
        return DefenseGanDefense(gan, 200, 10)
