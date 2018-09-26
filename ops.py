from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def layer_norm(inputs, data_format, training):

    return tf.contrib.layers.layer_norm(
        inputs=inputs,
        center=True,
        scale=True
    )


def instance_norm(inputs, data_format, training):

    return tf.contrib.layers.instance_norm(
        inputs=inputs,
        center=True,
        scale=True,
        data_format="NCHW" if data_format == "channels_first" else "NHWC"
    )


def batch_norm(inputs, data_format, training):

    return tf.contrib.layers.batch_norm(
        inputs=inputs,
        center=True,
        scale=True,
        is_training=training,
        data_format="NCHW" if data_format == "channels_first" else "NHWC"
    )


def upsampling2d(inputs, size, data_format):

    return tf.keras.layers.UpSampling2D(
        size=size,
        data_format=data_format
    )(inputs)


def dense_block(inputs, units, normalization, activation, data_format, training):

    inputs = tf.layers.dense(
        inputs=inputs,
        units=units,
        kernel_initializer=tf.variance_scaling_initializer()
    )

    if normalization:

        inputs = normalization(inputs, data_format, training)

    if activation:

        inputs = activation(inputs)

    return inputs


def conv2d_block(inputs, filters, kernel_size, strides, normalization, activation, data_format, training):

    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        data_format=data_format,
        kernel_initializer=tf.variance_scaling_initializer()
    )

    if normalization:

        inputs = normalization(inputs, data_format, training)

    if activation:

        inputs = activation(inputs)

    return inputs


def deconv2d_block(inputs, filters, kernel_size, strides, normalization, activation, data_format, training):

    inputs = tf.layers.conv2d_transpose(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        data_format=data_format,
        kernel_initializer=tf.variance_scaling_initializer(),
    )

    if normalization:

        inputs = normalization(inputs, data_format, training)

    if activation:

        inputs = activation(inputs)

    return inputs


def residual_block(inputs, filters, strides, normalization, activation, data_format, training):

    shortcut = conv2d_block(
        inputs=inputs,
        filters=filters,
        kernel_size=1,
        strides=strides,
        normalization=normalization,
        activation=None,
        data_format=data_format,
        training=training
    )

    inputs = conv2d_block(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=strides,
        normalization=normalization,
        activation=activation,
        data_format=data_format,
        training=training
    )

    inputs = conv2d_block(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=1,
        normalization=normalization,
        activation=None,
        data_format=data_format,
        training=training
    )

    inputs += shortcut

    inputs = activation(inputs)

    return inputs
