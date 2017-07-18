import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.layers as tflayers
from tfext.convnet import Convnet


class FcConvnetV2(Convnet):
    """
    Network that have the same structure as Alexnet + optional batchnorm.
    WARNING! You should feed images in HxWxC BGR format!
    """

    class RandomInitType:
        GAUSSIAN = 0,
        XAVIER_UNIFORM = 1,
        XAVIER_GAUSSIAN = 2

    def __init__(self,
                 im_shape=(227, 227, 3),
                 num_classes=1,
                 device_id='/gpu:0',
                 use_batch_norm=True,
                 random_init_type=RandomInitType.XAVIER_GAUSSIAN,
                 gpu_memory_fraction=None, **params):
        """
         Args:
          gpu_memory_fraction: Fraction on the max GPU memory to allocate for process needs.
            Allow auto growth if None (can take up to the totality of the memory).
        """
        self.input_shape = im_shape
        self.device_id = device_id
        self.random_init_type = random_init_type
        self.batch_norm_decay = 0.99

        if len(self.input_shape) == 2:
            self.input_shape += (3,)

        assert len(self.input_shape) == 3

        self.global_iter_counter = tf.Variable(0, name='global_iter_counter', trainable=False)
        with tf.variable_scope('input'):
            self.x = tf.placeholder(tf.float32, (None,) + self.input_shape, name='x')
            self.y_gt = tf.placeholder(tf.int32, shape=(None,), name='y_gt')
            self.is_phase_train = tf.placeholder(tf.bool, shape=tuple(), name='is_phase_train')
            self.dropout_keep_prob = tf.placeholder_with_default(1.0, tuple(),
                                                                 name='dropout_keep_prob')

        print 'Using Batch normalization after each conv layer:', use_batch_norm
        with tf.device(self.device_id):
            conv1 = self.conv_relu(self.x, kernel_size=11,
                                   kernels_num=96, stride=4,
                                   batch_norm=use_batch_norm,
                                   name='conv1')
            if not use_batch_norm:
                conv1 = tf.nn.local_response_normalization(conv1, depth_radius=2,
                                                                  alpha=2e-05,
                                                                  beta=0.75,
                                                                  bias=1.0)
            maxpool1 = tf.nn.max_pool(conv1,
                                      ksize=[1, 3, 3, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='VALID',
                                      name='maxpool1')

            conv2 = self.conv_relu(maxpool1, kernel_size=5,
                                   kernels_num=256, stride=1,
                                   group=2,
                                   batch_norm=use_batch_norm,
                                   name='conv2')
            if not use_batch_norm:
                conv2 = tf.nn.local_response_normalization(conv2, depth_radius=2,
                                                                  alpha=2e-05,
                                                                  beta=0.75,
                                                                  bias=1.0)
            maxpool2 = tf.nn.max_pool(conv2,
                                      ksize=[1, 3, 3, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='VALID',
                                      name='maxpool2')

            conv3 = self.conv_relu(maxpool2, kernel_size=3,
                                   kernels_num=384, stride=1,
                                   batch_norm=use_batch_norm,
                                   name='conv3')
            conv4 = self.conv_relu(conv3, kernel_size=3,
                                   kernels_num=384, stride=1,
                                   group=2,
                                   batch_norm=use_batch_norm,
                                   name='conv4')
            self.conv5 = self.conv_relu(conv4, kernel_size=3,
                                        kernels_num=256, stride=1,
                                        group=2,
                                        batch_norm=use_batch_norm,
                                        name='conv5')
            self.maxpool5 = tf.nn.max_pool(self.conv5,
                                           ksize=[1, 3, 3, 1],
                                           strides=[1, 2, 2, 1],
                                           padding='VALID',
                                           name='maxpool5')

            self.fc6, self.fc6_relu = self.fc_relu(self.maxpool5,
                                    num_outputs=4096,
                                    relu=True,
                                    batch_norm=use_batch_norm,
                                    weight_std=0.005, bias_init_value=0.1,
                                    name='fc6')

            dropout6 = tf.nn.dropout(self.fc6_relu, self.dropout_keep_prob, name='dropout6')

            self.fc7, fc7_relu = self.fc_relu(dropout6,
                                              num_outputs=4096,
                                              relu=True,
                                              batch_norm=use_batch_norm,
                                              weight_std=0.005, bias_init_value=0.1,
                                              name='fc7')
            self.fc7_dropout = tf.nn.dropout(fc7_relu, self.dropout_keep_prob, name='dropout7')

            self.fc8 = self.fc_relu(self.fc7_dropout,
                                    num_outputs=num_classes,
                                    relu=False,
                                    batch_norm=False,
                                    weight_std=0.01, bias_init_value=0.0,
                                    name='fc8')[0]
        self.logits = self.fc8

        self.graph = tf.get_default_graph()
        assert not use_batch_norm or len(self.graph.get_collection(
            tf.GraphKeys.UPDATE_OPS)) > 0, 'Must contain batch normalization update ops!'

        config = tf.ConfigProto(log_device_placement=False,
                                allow_soft_placement=True)
        # please do not use the totality of the GPU memory.
        if gpu_memory_fraction is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        self.sess = tf.Session(config=config)

    def restore_from_snapshot(self, snapshot_path, num_layers, restore_iter_counter=False):
        """
        :param snapshot_path: path to the snapshot file
        :param num_layers: number layers to restore from the snapshot
                            (conv1 is the #1, fc8 is the #8)
        :param restore_iter_counter: if True restore global_iter_counter from the snapshot

        WARNING! A call of sess.run(tf.initialize_all_variables()) after restoring from snapshot
                 will overwrite all variables and set them to initial state.
                 Call restore_from_snapshot() only after sess.run(tf.initialize_all_variables())!
        """
        if num_layers != 7 and num_layers != 5:
            raise ValueError('You can restore only 5 or 7 layers')
        if num_layers == 0:
            print 'FcConvnetV2::Not restoring anything'
            return

        with self.graph.as_default():
            vars_to_restore = tf.get_collection(tf.GraphKeys.VARIABLES, "conv")
            if restore_iter_counter:
                vars_to_restore.append(self.global_iter_counter)
            if num_layers == 7:
                vars_to_restore += tf.get_collection(tf.GraphKeys.VARIABLES, "fc6") + \
                                   tf.get_collection(tf.GraphKeys.VARIABLES, "fc7")

            print 'FcConvnetV2::Restoring {} layers:'.format(num_layers), \
                [v.name for v in vars_to_restore]
            saver = tf.train.Saver(vars_to_restore)
            saver.restore(self.sess, snapshot_path)

    def restore_from_alexnet_snapshot(self, snapshot_path, num_layers):
        """
        :param snapshot_path: path to the snapshot file
        :param num_layers: number layers to restore from the snapshot
                            (conv1 is the #1, fc6 is the #6)
        :param restore_iter_counter: if True restore global_iter_counter from the snapshot

        WARNING! A call of sess.run(tf.initialize_all_variables()) after restoring from snapshot
                 will overwrite all variables and set them to initial state.
                 Call restore_from_snapshot() only after sess.run(tf.initialize_all_variables())!
        """
        if num_layers > 7 or num_layers < 0:
            raise ValueError('You can restore from 0 to 5 layers.')
        if num_layers == 0:
            print 'Not restoring anything'
            return
        print 'Restoring {} layers from Alexnet model'.format(num_layers)

        var_names_to_restore = ['conv{}/bias:0'.format(i) for i in xrange(1, min(6, num_layers + 1))] + \
                               ['conv{}/weight:0'.format(i) for i in xrange(1, min(6, num_layers + 1))]
        var_names_to_restore += ['fc{}/bias:0'.format(i) for i in xrange(6, min(9, num_layers + 1))] + \
                                ['fc{}/weight:0'.format(i) for i in xrange(6, min(9, num_layers + 1))]
        print 'Restoring:', var_names_to_restore
        with self.graph.as_default():
            vars_to_restore = [
                self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, var_name)[0] for
                                            var_name in var_names_to_restore]
            saver = tf.train.Saver(vars_to_restore)
            saver.restore(self.sess, snapshot_path)