import tensorflow as tf
import numpy as np

class DefGanModel():

    #https://www.tensorflow.org/guide/saved_model
    def __init__(self):
        sess =  tf.Session(graph=tf.Graph())
        tf.saved_model.loader.load(sess, ["serve"], "defenses/defensegan/pred_model")
        tf.get_default_graph()
        self.sess = sess

    def predict(self, images):
        predictions =  self.sess.run("Softmax:0", feed_dict={"images_pl:0": images, "labels_pl:0": np.zeros((1, 10))})
        return predictions

    def __del__(self):
        self.sess.close()