import torchvision.transforms as transforms
import torchvision
import torch

'''
helper for getting pyTorch dataloaders for cifar10 or mnist dataset
'''

classes_cifar10 = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'adversarial')

classes_mnist = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'adversarial')

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
def get_data_loaders_cifar10(batchsize = 256, tf = None):

    if tf == None:
        #tf = transform=transforms.Compose([transforms.RandomHorizontalFlip(),
         #                                   transforms.RandomCrop(32, 4),
          #                                  transforms.ToTensor()])
        tf = transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(root='classifiers/cifar10_classifier/data', train=True,
                                            download=True, transform=tf)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='classifiers/cifar10_classifier/data', train=False,
                                           download=True, transform=transforms.Compose([transforms.ToTensor()]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                             shuffle=True, num_workers=0)

    return trainloader, testloader

def get_data_loaders_mnist(batchsize = 256, tf = None):
    if tf == None:
        tf = transforms.ToTensor()

    trainset = torchvision.datasets.MNIST(root='classifiers/mnist_classifier/data', train=True,
                                            download=True, transform=tf)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.MNIST(root='classifiers/mnist_classifier/data', train=False,
                                           download=True, transform=tf)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                             shuffle=True, num_workers=4)

    return trainloader, testloader