from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import collections
import functools
import operator
import resnet


class Model(resnet.Model):

    """ implementation of DCGAN in TensorFlow

    [1] [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks]
        (https://arxiv.org/pdf/1511.06434.pdf) by Alec Radford, Luke Metz, and Soumith Chintala, Nov 2015.

    [2] [Which Training Methods for GANs do actually Converge?](https://arxiv.org/pdf/1801.04406.pdf)
        by Lars Mescheder, Andreas Geiger, and Sebastian Nowozin, Jul 2018.
    """

    # NO BATCH NORMALIZATION

    # generator側のresblockはbottleneck使うのは気持ち悪い
    # resnet basedなアーキテクチャのconv層はbias無しにするべき？

    # instance noise未実装
    # 分散の求め方が不明
    # 焼きなましで求めると書いてあるような気がするけどそもそも目的関数は何？

    class Generator(object):

        def __init__(self, image_size, filters, block_params, bottleneck, version, final_conv_param):

            self.image_size = image_size
            self.filters = filters
            self.block_params = block_params
            self.bottleneck = bottleneck
            self.version = version
            self.final_conv_param = final_conv_param

            self.block_fn = ((Model.bottleneck_block_v1 if self.version == 1 else Model.bottleneck_block_v2) if self.bottleneck else
                             (Model.building_block_v1 if self.version == 1 else Model.building_block_v2))

            self.projection_shortcut = Model.projection_shortcut

        def __call__(self, inputs, data_format, training, reuse=False):

            with tf.variable_scope("generator", reuse=reuse):

                initial_image_size = [size >> len(self.block_params) for size in self.image_size]

                inputs = tf.layers.dense(
                    inputs=inputs,
                    units=self.filters * functools.reduce(operator.mul, initial_image_size)
                )

                inputs = tf.reshape(
                    tensor=inputs,
                    shape=([-1] + [self.filters] + initial_image_size if data_format == "channels_first" else
                           [-1] + initial_image_size + [self.filters])
                )

                if self.version == 1:

                    inputs = tf.nn.leaky_relu(inputs)

                for i, block_param in enumerate(self.block_params):

                    inputs = Model.block_layer(
                        inputs=inputs,
                        block_fn=self.block_fn,
                        blocks=block_param.blocks,
                        filters=self.filters >> i,
                        strides=block_param.strides,
                        projection_shortcut=self.projection_shortcut,
                        data_format=data_format,
                        training=training
                    )

                    inputs = tf.keras.layers.UpSampling2D(
                        size=2,
                        data_format=data_format
                    )(inputs)

                if self.version == 2:

                    inputs = tf.nn.leaky_relu(inputs)

                inputs = tf.layers.conv2d(
                    inputs=inputs,
                    filters=3,
                    kernel_size=self.final_conv_param.kernel_size,
                    strides=self.final_conv_param.strides,
                    padding="same",
                    data_format=data_format,
                    kernel_initializer=tf.variance_scaling_initializer(),
                )

                inputs = tf.nn.tanh(inputs)

                return inputs

    class Discriminator(object):

        def __init__(self, filters, initial_conv_param, block_params, bottleneck, version):

            self.filters = filters
            self.initial_conv_param = initial_conv_param
            self.block_params = block_params
            self.bottleneck = bottleneck
            self.version = version

            self.block_fn = ((Model.bottleneck_block_v1 if self.version == 1 else Model.bottleneck_block_v2) if self.bottleneck else
                             (Model.building_block_v1 if self.version == 1 else Model.building_block_v2))

            self.projection_shortcut = Model.projection_shortcut

        def __call__(self, inputs, data_format, training, reuse=False):

            with tf.variable_scope("discriminator", reuse=reuse):

                inputs = tf.layers.conv2d(
                    inputs=inputs,
                    filters=self.filters,
                    kernel_size=self.initial_conv_param.kernel_size,
                    strides=self.initial_conv_param.strides,
                    padding="same",
                    data_format=data_format,
                    use_bias=False,
                    kernel_initializer=tf.variance_scaling_initializer(),
                )

                if self.version == 1:

                    inputs = tf.nn.leaky_relu(inputs)

                for i, block_param in enumerate(self.block_params):

                    inputs = Model.block_layer(
                        inputs=inputs,
                        block_fn=self.block_fn,
                        blocks=block_param.blocks,
                        filters=self.filters << i,
                        strides=block_param.strides,
                        projection_shortcut=self.projection_shortcut,
                        data_format=data_format,
                        training=training
                    )

                    inputs = tf.layers.average_pooling2d(
                        inputs=inputs,
                        pool_size=2,
                        strides=2,
                        padding="same",
                        data_format=data_format
                    )

                if self.version == 2:

                    inputs = tf.nn.leaky_relu(inputs)

                inputs = tf.layers.flatten(inputs)

                inputs = tf.layers.dense(
                    inputs=inputs,
                    units=1
                )

                return inputs

    @staticmethod
    def building_block_v1(inputs, filters, strides, projection_shortcut, data_format, training):

        shortcut = inputs

        if projection_shortcut:

            shortcut = projection_shortcut(
                inputs=inputs,
                filters=filters,
                strides=strides,
                data_format=data_format
            )

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=3,
            strides=strides,
            padding="same",
            data_format=data_format,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
        )

        inputs = tf.nn.leaky_relu(inputs)

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            data_format=data_format,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
        )

        inputs += shortcut

        inputs = tf.nn.leaky_relu(inputs)

        return inputs

    @staticmethod
    def building_block_v2(inputs, filters, strides, projection_shortcut, data_format, training):

        shortcut = inputs

        inputs = tf.nn.leaky_relu(inputs)

        if projection_shortcut:

            shortcut = projection_shortcut(
                inputs=inputs,
                filters=filters,
                strides=strides,
                data_format=data_format
            )

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=3,
            strides=strides,
            padding="same",
            data_format=data_format,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
        )

        inputs = tf.nn.leaky_relu(inputs)

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            data_format=data_format,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
        )

        inputs += shortcut

        return inputs

    @staticmethod
    def bottleneck_block_v1(inputs, filters, strides, projection_shortcut, data_format, training):

        shortcut = inputs

        if projection_shortcut:

            shortcut = projection_shortcut(
                inputs=inputs,
                filters=filters << 2,
                strides=strides,
                data_format=data_format
            )

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=1,
            strides=1,
            padding="same",
            data_format=data_format,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
        )

        inputs = tf.nn.leaky_relu(inputs)

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=3,
            strides=strides,
            padding="same",
            data_format=data_format,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
        )

        inputs = tf.nn.leaky_relu(inputs)

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters << 2,
            kernel_size=1,
            strides=1,
            padding="same",
            data_format=data_format,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
        )

        inputs += shortcut

        inputs = tf.nn.leaky_relu(inputs)

        return inputs

    @staticmethod
    def bottleneck_block_v2(inputs, filters, strides, projection_shortcut, data_format, training):

        shortcut = inputs

        inputs = tf.nn.leaky_relu(inputs)

        if projection_shortcut:

            shortcut = projection_shortcut(
                inputs=inputs,
                filters=filters << 2,
                strides=strides,
                data_format=data_format
            )

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=1,
            strides=1,
            padding="same",
            data_format=data_format,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
        )

        inputs = tf.nn.leaky_relu(inputs)

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=3,
            strides=strides,
            padding="same",
            data_format=data_format,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
        )

        inputs = tf.nn.leaky_relu(inputs)

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters << 2,
            kernel_size=1,
            strides=1,
            padding="same",
            data_format=data_format,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
        )

        inputs += shortcut

        return inputs
