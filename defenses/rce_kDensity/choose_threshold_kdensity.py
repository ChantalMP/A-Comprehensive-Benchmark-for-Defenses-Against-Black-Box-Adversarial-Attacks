from utils.data_handling import get_data_loaders_cifar10, get_data_loaders_mnist
from utils.config_loader import load_config
from classifiers.mymodel import MyModel
from utils.modelWrapper import ModelWrapper
from torch.autograd import Variable
from torchvision import transforms
import pickle
import torch
import statistics
import numpy as np
from defenses.rce_kDensity.rce_kDensity import RCE_KDensity
from utils.image_utils import plot_image

'''
class to calculate the average k density scores of adversarial(FGSM) and non-adversarial(clean) data
needed to set threshold for our attack
'''
def test_images(testdata, dataset):
    labels = np.load("defenses/rce_kDensity/data/labels_{}.npy".format(dataset))
    pred, hidden_features = model.batch_predictions(testdata.cpu().numpy(), detect=False)
    scores = detector.detect(hidden_features, pred)
    print("acc: {}".format(np.sum(np.argmax(labels, axis=1)==np.argmax(pred, axis=1))/len(labels)))
    return scores

def test_data_loader(testdata, net, noise = False):
    score_avg_sum = 0
    i=0
    scores = None
    for data in testdata:
        images, labels = data
        #Note: to run this in batch_predictions return hidden_features as well
        pred, hidden_features = model.batch_predictions(images.cpu().numpy(), detect= False)
        if scores is None:
            scores = detector.detect(hidden_features, pred)
        else:
            scores = np.append(scores, detector.detect(hidden_features, pred))
        i+=1
        if i == 5:
            break
    return scores



if __name__ == '__main__':

    config = load_config("config_files/cifar10_RCE_noDetector.yaml")
    dataset = config['dataset']
    detector = config['detector']
    classifier_name = config['classifier_name']
    classifier_path = "classifiers/{}_classifier/models/{}".format(dataset, classifier_name)
    model_type = config['model_type']

    classifier = RCE_KDensity("rce", dataset=dataset)
    model = ModelWrapper(model_type, classifier, detector=detector, dataset=dataset)
    if dataset == 'cifar10':
        train_loader, _ = get_data_loaders_cifar10(batchsize=100) #tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]))
        dataset = "cifar"
    if dataset == 'mnist':
        train_loader, _ = get_data_loaders_mnist(batchsize=100)
    detector = classifier

    print("trainingdata")
    train = test_data_loader(train_loader, classifier)
    print("avg: {}".format(np.average(train)))

    #clean_img = torch.from_numpy(np.load("defenses/rce_kDensity/data/Clean_mnist_fgsm.npy")).transpose(3, 1).transpose(2,3)
    clean_img = torch.from_numpy(np.load("defenses/rce_kDensity/data/Clean_cifar_fgsm.npy")).transpose(3, 1).transpose(2,3)
    print("clean")
    clean = test_images(clean_img, dataset)
    print("avg: {}".format(np.average(clean)))


    #adv_img = torch.from_numpy(np.load("defenses/rce_kDensity/data/Adv_mnist_fgsm.npy")).transpose(3,1).transpose(2,3)
    adv_img = torch.from_numpy(np.load("defenses/rce_kDensity/data/Adv_cifar_fgsm.npy")).transpose(3,1).transpose(2,3)


    print("adv")
    adv = test_images(adv_img, dataset)
    print("avg: {}".format(np.average(adv)))
    print("finished")









