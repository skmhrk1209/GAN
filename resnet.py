from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import collections
import ops


class Generator(object):

    ResidualParam = collections.namedtuple("ResidualParam", ("filters", "blocks"))

    def __init__(self, image_size, filters, residual_params, data_format):

        self.image_size = image_size
        self.filters = filters
        self.residual_params = residual_params
        self.data_format = data_format

    def __call__(self, inputs, training, name="generator", reuse=False):

        with tf.variable_scope(name, reuse=reuse):

            seed_size = (np.array(self.image_size) >> len(self.residual_params)).tolist()

            inputs = ops.dense_block(
                inputs=inputs,
                units=np.prod(seed_size) * self.filters,
                normalization=None,
                activation=tf.nn.leaky_relu,
                data_format=self.data_format,
                training=training
            )

            inputs = tf.reshape(
                tensor=inputs,
                shape=[-1] + seed_size + [self.filters]
            )

            if self.data_format == "channels_first":

                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            for residual_param in self.residual_params:

                inputs = ops.upsampling2d(
                    inputs=inputs,
                    size=2,
                    data_format=self.data_format
                )

                for _ in range(residual_param.blocks):

                    inputs = ops.residual_block(
                        inputs=inputs,
                        filters=residual_param.filters,
                        strides=1,
                        normalization=None,
                        activation=tf.nn.leaky_relu,
                        data_format=self.data_format,
                        training=training
                    )

            inputs = ops.conv2d_block(
                inputs=inputs,
                filters=3,
                kernel_size=3,
                strides=1,
                normalization=None,
                activation=tf.nn.sigmoid,
                data_format=self.data_format,
                training=training
            )

            return inputs


class Discriminator(object):

    ResidualParam = collections.namedtuple("ResidualParam", ("filters", "blocks"))

    def __init__(self, filters, residual_params, data_format):

        self.filters = filters
        self.residual_params = residual_params
        self.data_format = data_format

    def __call__(self, inputs, training, name="discriminator", reuse=False):

        with tf.variable_scope(name, reuse=reuse):

            inputs = ops.conv2d_block(
                inputs=inputs,
                filters=self.filters,
                kernel_size=3,
                strides=1,
                normalization=None,
                activation=tf.nn.leaky_relu,
                data_format=self.data_format,
                training=training
            )

            for residual_param in self.residual_params:

                for _ in range(residual_param.blocks):

                    inputs = ops.residual_block(
                        inputs=inputs,
                        filters=residual_param.filters,
                        strides=1,
                        normalization=None,
                        activation=tf.nn.leaky_relu,
                        data_format=self.data_format,
                        training=training
                    )

                inputs = tf.layers.average_pooling2d(
                    inputs=inputs,
                    pool_size=2,
                    strides=2,
                    padding="same",
                    data_format=self.data_format
                )

            inputs = tf.layers.flatten(inputs)

            inputs = ops.dense_block(
                inputs=inputs,
                units=1,
                normalization=None,
                activation=None,
                data_format=self.data_format,
                training=training
            )

            return inputs
