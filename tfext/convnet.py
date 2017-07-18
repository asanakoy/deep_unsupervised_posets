################################################################################
# All conv net
# Copyright (c) 2016 Artsiom Sanakoyeu
################################################################################
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tflayers


class Convnet(object):
    """
    Network that have the same structure as first 5 layers in Alexnet.
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
                 random_init_type=RandomInitType.XAVIER_GAUSSIAN,
                 gpu_memory_fraction=None, **params):
        """
         Args:
          init_model: dict containing network weights, or a string with path to .np file with the dict,
            if is None then init using random weights and biases
          num_classes: number of output classes
          gpu_memory_fraction: Fraction on the max GPU memory to allocate for process needs.
            Allow auto growth if None (can take up to the totality of the memory).
        :return:
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

        with tf.device(self.device_id):
            conv1 = self.conv_relu(self.x, kernel_size=11,
                                   kernels_num=96, stride=4,
                                   name='conv1')
            maxpool1 = tf.nn.max_pool(conv1,
                                      ksize=[1, 3, 3, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='VALID',
                                      name='maxpool1')

            conv2 = self.conv_relu(maxpool1, kernel_size=5,
                                   kernels_num=256, stride=1,
                                   group=2,
                                   name='conv2')
            maxpool2 = tf.nn.max_pool(conv2,
                                      ksize=[1, 3, 3, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='VALID',
                                      name='maxpool2')

            conv3 = self.conv_relu(maxpool2, kernel_size=3,
                                   kernels_num=384, stride=1,
                                   name='conv3')
            conv4 = self.conv_relu(conv3, kernel_size=3,
                                   kernels_num=384, stride=1,
                                   group=2,
                                   name='conv4')
            self.conv5 = self.conv_relu(conv4, kernel_size=3,
                                   kernels_num=256, stride=1,
                                   group=2,
                                   name='conv5')
            self.maxpool5 = tf.nn.max_pool(self.conv5,
                                       ksize=[1, 3, 3, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='VALID',
                                       name='maxpool5')
            dropout5 = tf.nn.dropout(self.maxpool5, self.dropout_keep_prob, name='dropout5')

            self.fc6 = self.fc_relu(dropout5,
                                     num_outputs=num_classes,
                                     relu=False,
                                     weight_std=0.01, bias_init_value=0.0,
                                     name='fc6')[0]
            self.logits = self.fc6

            with tf.variable_scope('output'):
                self.prob = tf.nn.softmax(self.fc6, name='prob')

            fc6_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc6/weight:0")[0]
            fc6_b = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc6/bias:0")[0]
            self.reset_fc6_op = tf.variables_initializer([fc6_w, fc6_b], name='reset_fc6')

        self.graph = tf.get_default_graph()
        assert len(self.graph.get_collection(tf.GraphKeys.UPDATE_OPS)) > 0, 'Must contain batch normalization update ops!'

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
        if num_layers > 6 or num_layers < 5:
            raise ValueError('You can restore only 5 or 6 layers.')
        if num_layers == 0:
            print 'Convnet::Not restoring anything'
            return

        with self.graph.as_default():
            if num_layers == 5:
                self.reset_fc6()
                fc6_vars = tf.get_collection(tf.GraphKeys.VARIABLES, "fc6")
                vars_to_restore = tf.get_collection(tf.GraphKeys.VARIABLES)
                vars_to_restore = [x for x in vars_to_restore if x not in fc6_vars]
                vars_to_restore = [x for x in vars_to_restore if x != self.global_iter_counter]
                if restore_iter_counter:
                    vars_to_restore.append(self.global_iter_counter)
                print 'Convnet::Restoring 5 layers:', [v.name for v in vars_to_restore]
                saver = tf.train.Saver(vars_to_restore)
            else:
                print 'Convnet::Restoring Everything (6 layers)'
                saver = tf.train.Saver()
                if not restore_iter_counter:
                    tf.variables_initializer(self.global_iter_counter)
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
        if num_layers > 5 or num_layers < 0:
            raise ValueError('You can restore from 0 to 5 layers.')
        if num_layers == 0:
            return
        print 'Restoring {} layers from Alexnet model'.format(num_layers)

        var_names_to_restore = ['conv{}/bias:0'.format(i) for i in xrange(1, min(6, num_layers + 1))] + \
                               ['conv{}/weight:0'.format(i) for i in xrange(1, min(6, num_layers + 1))]
        with self.graph.as_default():
            vars_to_restore = [self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, var_name)[0] for var_name in var_names_to_restore]
            saver = tf.train.Saver(vars_to_restore)
            saver.restore(self.sess, snapshot_path)

    def reset_fc6(self):
        print 'Resetting fc6 to random'
        self.sess.run(self.reset_fc6_op)

    def get_conv_weights(self, kernel_size, num_input_channels, kernels_num,
                         weight_std=0.01, bias_init_value=0.1):
        w = self.random_weight_variable((kernel_size, kernel_size,
                                         num_input_channels,
                                         kernels_num),
                                        stddev=weight_std)
        b = self.random_bias_variable((kernels_num,), value=bias_init_value)
        return w, b

    def get_fc_weights(self, num_inputs, num_outputs, weight_std=0.005, bias_init_value=0.1):
        w = self.random_weight_variable((num_inputs, num_outputs), stddev=weight_std)
        b = self.random_bias_variable((num_outputs,), value=bias_init_value)
        return w, b

    def random_weight_variable(self, shape, stddev=0.01):
        """
        stddev is used only for RandomInitType.GAUSSIAN
        """
        if self.random_init_type == Convnet.RandomInitType.GAUSSIAN:
            initial = tf.truncated_normal(shape, stddev=stddev)
            return tf.Variable(initial, name='weight')
        elif self.random_init_type == Convnet.RandomInitType.XAVIER_GAUSSIAN:
            return tf.get_variable("weight", shape=shape,
                                   initializer=tf.contrib.layers.xavier_initializer(
                                       uniform=False))
        elif self.random_init_type == Convnet.RandomInitType.XAVIER_UNIFORM:
            return tf.get_variable("weight", shape=shape,
                                   initializer=tf.contrib.layers.xavier_initializer(
                                       uniform=True))
        else:
            raise ValueError('Unknown random_init_type')

    @staticmethod
    def random_bias_variable(shape, value=0.1):
        initial = tf.constant(value, shape=shape)
        return tf.Variable(initial, name='bias')

    @staticmethod
    def conv(input_tensor, kernel, biases, stride, padding="VALID", group=1):

        c_i = input_tensor.get_shape()[-1]
        assert c_i % group == 0
        assert kernel.get_shape()[3] % group == 0

        def convolve(inp, w, name=None):
            return tf.nn.conv2d(inp, w, [1, stride, stride, 1], padding=padding, name=name)

        if group == 1:
            conv = convolve(input_tensor, kernel)
        else:
            input_groups = tf.split(axis=3, num_or_size_splits=group, value=input_tensor)
            kernel_groups = tf.split(axis=3, num_or_size_splits=group, value=kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            conv = tf.concat(axis=3, values=output_groups)
        # TODO: no need to reshape?
        return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:],
                          name='conv')

    def conv_relu(self, input_tensor, kernel_size, kernels_num, stride, batch_norm=True,
                  group=1, name=None):
        with tf.variable_scope(name) as scope:
            assert int(input_tensor.get_shape()[3]) % group == 0
            num_input_channels = int(input_tensor.get_shape()[3]) / group
            w, b = self.get_conv_weights(kernel_size, num_input_channels, kernels_num)
            conv = Convnet.conv(input_tensor, w, b, stride, padding="SAME", group=group)
            if batch_norm:
                conv = tf.cond(self.is_phase_train,
                               lambda: tflayers.batch_norm(conv,
                                                           decay=self.batch_norm_decay,
                                                           is_training=True,
                                                           trainable=True,
                                                           reuse=None,
                                                           scope=scope),
                               lambda: tflayers.batch_norm(conv,
                                                           decay=self.batch_norm_decay,
                                                           is_training=False,
                                                           trainable=True,
                                                           reuse=True,
                                                           scope=scope))
            conv = tf.nn.relu(conv, name=name)
        return conv

    def fc_relu(self, input_tensor, num_outputs, relu=False, batch_norm=False, weight_std=0.005,
                bias_init_value=0.1, name=None):
        if batch_norm and not relu:
            raise ValueError('Cannot use batch normalization without following RELU')
        with tf.variable_scope(name) as scope:
            num_inputs = int(np.prod(input_tensor.get_shape()[1:]))
            w, b = self.get_fc_weights(num_inputs, num_outputs,
                                       weight_std=weight_std,
                                       bias_init_value=bias_init_value)

            fc_relu = None
            input_tensor_reshaped = tf.reshape(input_tensor, [-1, num_inputs])
            fc = tf.add(tf.matmul(input_tensor_reshaped, w), b, name='fc' if relu or batch_norm else name)
            if batch_norm:
                fc = tf.cond(self.is_phase_train,
                             lambda: tflayers.batch_norm(fc,
                                                           decay=self.batch_norm_decay,
                                                           is_training=True,
                                                           trainable=True,
                                                           reuse=None,
                                                           scope=scope),
                              lambda: tflayers.batch_norm(fc,
                                                           decay=self.batch_norm_decay,
                                                           is_training=False,
                                                           trainable=True,
                                                           reuse=True,
                                                           scope=scope))
            if relu:
                fc_relu = tf.nn.relu(fc, name=name)
        return fc, fc_relu
