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
from defenses.mahalanobis.mahalanobis import MahalanobisDetector

'''
class to calculate the average logistic regressor predictions using mahalanobis scores of adversarial(FGSM) and non-adversarial(clean) data
needed to set threshold for our attack
'''
def test_images(testdata):
    reg_scores = detector.detect(testdata, np.ones(testdata.size(0)))
    print("avg: {}".format(np.average(reg_scores)))

def test_data_loader(testdata, net, noise = False):
    net.eval()
    score_avg_sum = 0
    i=0
    for data in testdata:
        images, labels = data
        labels = model.batch_predictions(images.cpu().numpy(), detect= False)
        if noise:
            images = torch.rand(images.shape)
        scores = detector.detect(images, labels)
        score_avg = statistics.mean(scores)
        score_avg_sum += score_avg
        i+=1
        if i == 3:
            break
    print("avg: {}".format(np.average(score_avg_sum/i)))



if __name__ == '__main__':

    config = load_config("config_files/cifar10_mahalanobis.yaml")
    dataset = config['dataset']
    detector = config['detector']
    classifier_name = config['classifier_name']
    classifier_path = "classifiers/{}_classifier/models/{}".format(dataset, classifier_name)

    classifier, _, _, _, _ = MyModel(dataset).get_model(model_name=classifier_name, model_path=classifier_path)
    model = ModelWrapper("PyTorch", classifier, detector=detector, dataset=dataset)
    if dataset == 'cifar10':
        train_loader, _ = get_data_loaders_cifar10(batchsize=100, tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]))
    if dataset == 'mnist':
        train_loader, _ = get_data_loaders_mnist(batchsize=100, tf=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), ]))
    test_clean_data = torch.load('defenses/mahalanobis/adv_output/resnet_{}/clean_data_resnet_{}_FGSM.pth'.format(dataset, dataset))
    test_adv_data = torch.load('defenses/mahalanobis/adv_output/resnet_{}/adv_data_resnet_{}_FGSM.pth'.format(dataset, dataset))
    test_noisy_data = torch.load('defenses/mahalanobis/adv_output/resnet_{}/noisy_data_resnet_{}_FGSM.pth'.format(dataset, dataset))
    test_label = torch.load('defenses/mahalanobis/adv_output/resnet_{}/label_resnet_{}_FGSM.pth'.format(dataset, dataset))
    detector = MahalanobisDetector(detector[1], classifier, dataset)
    print("trainingdata noise")
    test_data_loader(train_loader, classifier, noise=True)
    print("trainingdata")
    test_data_loader(train_loader, classifier)
    print("clean")
    test_images(test_clean_data)
    print("adv")
    test_images(test_adv_data)
    print("noisy")
    test_images(test_noisy_data)








