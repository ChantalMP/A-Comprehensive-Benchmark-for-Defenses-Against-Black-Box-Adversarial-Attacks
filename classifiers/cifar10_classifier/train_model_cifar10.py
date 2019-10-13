from utils.data_handling import get_data_loaders_cifar10
from classifiers.mymodel import MyModel

"""train new cifar10 classifier (resnet)
"""
if __name__ == '__main__':
    train_iterations = 20
    epochs = 10
    model_name = "resnet44_200epochs.pt"
    trainloader, testloader = get_data_loaders_cifar10()
    model = MyModel("cifar10")
    net, criterion, optimizer, lr_scheduler, start_epoch = model.get_model(model_name)
    for i in range(train_iterations):
        save_name = model.train(trainloader, net, criterion, optimizer, epochs, start_epoch, model_name, lr_scheduler)
        model.test(testloader, net, "{}_epochs_{}".format(save_name,epochs+start_epoch))
        start_epoch += 10