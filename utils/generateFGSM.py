# import foolbox
# import tensorflow as tf
# from utils.data_handling import get_data_loaders_cifar10, get_data_loaders_mnist
#
#
# tf.reset_default_graph()
# images = images - 0.5
# model = resnet_model_mnist.ResNet(self.hps, images, 'eval', Reuse=False)
# model_name = "defenses/rce_kDensity/models_{}/resnet32_{}/model.ckpt-20001".format(
#     self.dataset, self.classifier_name)
#
#
# with foolbox.models.TensorFlowModel(images, logits, (0, 255)) as model:
#     restorer.restore(model.session, '/path/to/vgg_19.ckpt')
#     print(np.argmax(model.forward_one(image)))
#
#
#
# train_loader, _ = get_data_loaders_mnist(batchsize=100)
