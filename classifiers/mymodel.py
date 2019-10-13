import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
from utils.data_handling import classes_cifar10, classes_mnist
from classifiers.resnet import resnet44
import torch.nn.parallel
import torch.utils.data
from defenses.mahalanobis import models
from keras.models import model_from_json


cuda = True
batchsize=128

class MyModel:

    """defines my resnet classifier for either mnist or cifar10 classification
       dataset: string describing used dataset: either "mnist" or "cifar10"
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.pathPrefix = "classifiers/{}_classifier/".format(self.dataset)
        self.classes = classes_cifar10 if self.dataset == "cifar10" else classes_mnist

    """
    returns either model saved at given model_path or initializes new resnet
    model_name: name under which model should be saved
    """
    def get_model(self, model_name = "", model_path = "", model_type = "PyTorch"):
        start_epoch = 0
        if model_name == "mahalanobis_cifar10":
            pre_trained_net = 'defenses/mahalanobis/pre_trained/resnet_cifar10.pth'
            net = models.ResNet34(num_c=10, dataset=self.dataset)
            net.load_state_dict(torch.load(pre_trained_net, map_location="cuda:0"))
        elif model_name == "modelA_ens" or model_name == "modelA":
            net = self.load_ens_model(model_name)
        else:
            self.save_name = model_name
            if model_name == "state_epoch11_mahalanobis_mnist.pt":
                net = models.ResNet34(num_c=10, dataset=self.dataset)
            else:
                net = torch.nn.DataParallel(resnet44(self.dataset))
            if (model_name != ""):
                if os.path.isfile(model_path):
                    print("=> loading model '{}'".format(model_name))
                    checkpoint = torch.load(model_path)
                    start_epoch = checkpoint['epochs']
                    net.load_state_dict(checkpoint['state_dict'])
                else:
                    print("did not find model with name {}, starting from scratch!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(model_name))

        criterion, optimizer, lr_scheduler = None,None,None
        if model_type == "PyTorch":
            criterion = nn.CrossEntropyLoss()
            if cuda:
                net.cuda()
                criterion.cuda()
            optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

            lr_scheduler = 0 # TODO: repair
            #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
              #  milestones=[100, 150], last_epoch=start_epoch-1)

        return net, criterion, optimizer, lr_scheduler, start_epoch

    def load_ens_model(self, model_name):
        try:
            with open("classifiers/mnist_classifier/models/{}.json".format(model_name), 'r') as f:
                json_string = f.read()
                model = model_from_json(json_string)
        except IOError:
            print("ensemble model could not be loaded")

        model.load_weights("classifiers/mnist_classifier/models/{}".format(model_name))
        print("loaded model {}".format(model_name))
        return model
    """
    model training function
    trainloader: dataloader with training data
    net: classifier to train
    criterion, optimizer, lr_scheduler: training hyperparameters
    epochs: training epochs
    start_epochs: epoch where to resume training (0 if training from scratch)
    net_name: save name of classifier
    """
    def train(self, trainloader, net, criterion, optimizer, epochs, start_epoch, net_name, lr_scheduler):
        global pathPrefix
        net.train()

        if not os.path.isdir(self.pathPrefix+"models"):
            os.mkdir(self.pathPrefix+"models")

        for epoch in range(start_epoch+1, epochs+start_epoch+1):  # loop over the dataset multiple times

            running_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                if cuda:
                    inputs, labels = inputs.cuda(), Variable(labels.cuda())
                else:
                    inputs, labels = inputs, Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, pred = outputs.data.topk(1,1)
                correct += (pred.squeeze()==labels).sum()
                total += len(inputs)

                # save statistics
                running_loss += loss.data.item()
                if i % 45 == 0:  # print every 500 batches
                    print('{}, {} loss: {}'.format(epoch, i, running_loss / 100))
                    running_loss = 0.0
                    train_acc = correct*100/total
                    correct = 0
                    total = 0
                    print('train accuracy: {} %'.format(train_acc))

                    state = {
                        'state_dict': net.state_dict(),
                        'train_cc': train_acc,
                        'epochs': epoch
                    }
                    torch.save(state, "{}/models/state_epoch{}_{}".format(self.pathPrefix, epoch, net_name))
                    self.save_name = "state_epoch{}_{}".format(epoch, net_name)

            lr_scheduler.step()

        print('Finished Training')
        return self.save_name

    """
    model testing fuction
    testloader: dataloader with test data
    net: classifier to test
    modelname: save name of classifier
    """
    def test(self, testloader, net, modelname):
        net.eval()
        if not os.path.isdir(self.pathPrefix+"results"):
            os.mkdir(self.pathPrefix+"results")
        correct = 0
        total = 0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        for data in testloader:
            images, labels = data
            if cuda:
                outputs = net(Variable(images.cuda()))
            else:
                outputs = net(Variable(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if cuda:
                predicted = predicted.cuda()
                labels = labels.cuda()
            correct += (predicted == labels).sum()
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                if (c[i] == 1):
                    class_correct[label] += int(c[i])
                class_total[label] += 1

        file = open(self.pathPrefix+"results/" + modelname, "w")
        file.write('Accuracy of the network on the 10000 test images: {} \n'.format(
                100 * correct*1.0 / total*1.0))
        print('Accuracy of the network on the 10000 test images: {} \n'.format(
                100.0 * correct.item()*1.0 / total*1.0))
        print("results saved to: {}".format(modelname))
        for i in range(10):
            file.write('Accuracy of %5s : %2d %% \n' % (
                self.classes[i], 100 * class_correct[i] / class_total[i]))