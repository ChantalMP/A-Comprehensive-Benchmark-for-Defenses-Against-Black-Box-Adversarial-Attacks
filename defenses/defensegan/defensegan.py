import tensorflow as tf
import numpy as np
from mem_top import  mem_top
from pympler.tracker import SummaryTracker

class DefenseGanDefense():

    """defenseGan instance"""

    def __init__(self, gan, L, R):
        self.L = L
        self.R = R
        self.gan = gan
        self.gan.rec_rr = self.R
        self.gan.rec_iters = self.L
        self.rec_operations = {}

    """apply defense
       images: batch of images to recontruct with defenseGan
       returns reconstructed images which should be given to classifier
    """
    def defend(self, images):
        images = np.transpose(images, (0,2,3,1))
        sess = self.gan.sess
        batch_size = len(images)
        if batch_size in self.rec_operations:
            images_placeholder = self.rec_operations[batch_size][1]
            rec_op = self.rec_operations[batch_size][0]
        else:
            images_placeholder = tf.placeholder(tf.float32, shape=images.shape)
            rec_op = self.gan.reconstruct(images_placeholder, batch_size=batch_size, reconstructor_id=batch_size)
            self.rec_operations[batch_size] = (rec_op, images_placeholder)
            sess.run(tf.local_variables_initializer())

        images = sess.run(rec_op, feed_dict={images_placeholder:images})
        images = np.transpose(images, (0, 3, 1, 2))

        return images

