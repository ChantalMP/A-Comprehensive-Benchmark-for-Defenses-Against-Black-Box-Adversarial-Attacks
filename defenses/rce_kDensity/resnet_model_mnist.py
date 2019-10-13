from collections import namedtuple

import numpy as np
import tensorflow as tf
import six

from tensorflow.python.training import moving_averages


HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer,'
                     'RCE_train')


class ResNet(object):
  """ResNet model."""

  def __init__(self, hps, images, mode, labels=None, Reuse=False):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self._images = images
    self.labels = labels
    self.mode = mode
    self.Reuse = Reuse
    self._extra_train_ops = []

  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.contrib.framework.get_or_create_global_step()
    self._build_model()
    if self.mode == 'train':
      self._build_train_op()
      self.summaries = tf.summary.merge_all()

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    """Build the core model within the graph."""
    with tf.variable_scope('init',reuse=self.Reuse):
      x = self._images
      x = self._conv('init_conv', x, 3, 1, 16, self._stride_arr(1))

    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    if self.hps.use_bottleneck:
      res_func = self._bottleneck_residual
      filters = [16, 64, 128, 256]
    else:
      res_func = self._residual
      filters = [16, 16, 32, 64]
      # Uncomment the following codes to use w28-10 wide residual network.
      # It is more memory efficient than very deep residual network and has
      # comparably good performance.
      # https://arxiv.org/pdf/1605.07146v1.pdf
      # filters = [16, 160, 320, 640]
      # Update hps.num_residual_units to 4

    with tf.variable_scope('unit_1_0',reuse=self.Reuse):
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                   activate_before_residual[0])
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_1_%d' % i,reuse=self.Reuse):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

    with tf.variable_scope('unit_2_0',reuse=self.Reuse):
      x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                   activate_before_residual[1])
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_2_%d' % i,reuse=self.Reuse):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

    with tf.variable_scope('unit_3_0',reuse=self.Reuse):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_3_%d' % i,reuse=self.Reuse):
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

    with tf.variable_scope('unit_last',reuse=self.Reuse):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._global_avg_pool(x)
      self.t_SNE_logits = x

    with tf.variable_scope('logit',reuse=self.Reuse):
      logits = self._fully_connected(x, self.hps.num_classes)
      self.predictions = tf.nn.softmax(logits)
    if self.mode=='train':
      with tf.variable_scope('costs',reuse=self.Reuse):
        if self.hps.RCE_train:
          print('Using RCE as objective function')
          xent = (tf.reduce_sum(self.labels * tf.log(tf.nn.softmax(logits)), axis=1) - tf.reduce_sum(
            tf.log(tf.nn.softmax(logits)), axis=1)) / (self.hps.num_classes - 1)
        else:
          print('Using CE as objective function')
          xent = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.labels)
        self.cost = tf.reduce_mean(xent, name='xent')
        self.cost += self._decay()

        tf.summary.scalar('cost', self.cost)

  def _build_train_op(self):
    """Build training specific ops for the graph."""
    self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    tf.summary.scalar('learning_rate', self.lrn_rate)

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.cost, trainable_variables)

    if self.hps.optimizer == 'sgd':
      print('Optimizer is sgd')
      optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.hps.optimizer == 'mom':
      print('Optimizer is mom')
      optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
    elif self.hps.optimizer == 'adam':
      print('Optimizer is adam')
      optimizer = tf.train.AdamOptimizer(
        self.lrn_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0)
    elif self.hps.optimizer == 'rmsprop':
      print('Optimizer is rmsprop')
      optimizer = tf.train.RMSPropOptimizer(
        self.lrn_rate,
        decay=0.9,
        momentum=0.9,
        epsilon=1.0)
    else:
      print('Wrong optimizer')
      optimizer = None
    apply_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=self.global_step, name='train_step')

    train_ops = [apply_op] + self._extra_train_ops
    self.train_op = tf.group(*train_ops)

  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.variable_scope(name,reuse=self.Reuse):
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if self.mode == 'train':
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
        tf.summary.histogram(mean.op.name, mean)
        tf.summary.histogram(variance.op.name, variance)
      # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y

  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation',reuse=self.Reuse):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation',reuse=self.Reuse):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub1',reuse=self.Reuse):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2',reuse=self.Reuse):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add',reuse=self.Reuse):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x

  def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                           activate_before_residual=False):
    """Bottleneck residual unit with 3 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('common_bn_relu',reuse=self.Reuse):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_bn_relu',reuse=self.Reuse):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub1',reuse=self.Reuse):
      x = self._conv('conv1', x, 1, in_filter, out_filter/4, stride)

    with tf.variable_scope('sub2',reuse=self.Reuse):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter/4, out_filter/4, [1, 1, 1, 1])

    with tf.variable_scope('sub3',reuse=self.Reuse):
      x = self._batch_norm('bn3', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv3', x, 1, out_filter/4, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add',reuse=self.Reuse):
      if in_filter != out_filter:
        orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
      x += orig_x

    tf.logging.info('image after unit %s', x.get_shape())
    return x

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
        # tf.summary.histogram(var.op.name, var)

    return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name,reuse=self.Reuse):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.hps.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])
