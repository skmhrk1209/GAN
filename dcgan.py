from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import ops
import utils
import os
import itertools
import collections
import time
import cv2


class Model(object):

    """ implementation of DCGAN in TensorFlow

    [1] [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks]
        (https://arxiv.org/pdf/1511.06434.pdf) by Alec Radford, Luke Metz, and Soumith Chintala, Nov 2015.

    [2] [Which Training Methods for GANs do actually Converge?](https://arxiv.org/pdf/1801.04406.pdf)
        by Lars Mescheder, Andreas Geiger, and Sebastian Nowozin, Jul 2018.
    """

    BlockParam = collections.namedtuple("BlockParam", ("filters", "blocks"))
    GeneratorParam = collections.namedtuple("GeneratorParam", ("image_size", "filters", "block_params", "data_format"))
    DiscriminatorParam = collections.namedtuple("DiscriminatorParam", ("filters", "block_params", "data_format"))
    HyperParam = collections.namedtuple("HyperParam", ("latent_dimensions", "gradient_coefficient"))
    DatasetParam = collections.namedtuple("DatasetParam", ("filenames", "batch_size", "num_epochs", "buffer_size"))

    class Generator(object):

        def __init__(self, image_size, filters, block_params, data_format):

            self.image_size = image_size
            self.filters = filters
            self.block_params = block_params
            self.data_format = data_format

        def __call__(self, inputs, training, name="generator", reuse=False):

            with tf.variable_scope(name, reuse=reuse):

                seed_size = (np.array(self.image_size) >> len(self.block_params)).tolist()

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

                for block_param in self.block_params:

                    inputs = ops.upsampling2d(
                        inputs=inputs,
                        size=2,
                        data_format=self.data_format
                    )

                    for _ in range(block_param.blocks):

                        inputs = ops.residual_block(
                            inputs=inputs,
                            filters=block_param.filters,
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

        def __init__(self, filters, block_params, data_format):

            self.filters = filters
            self.block_params = block_params
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

                for block_param in self.block_params:

                    for _ in range(block_param.blocks):

                        inputs = ops.residual_block(
                            inputs=inputs,
                            filters=block_param.filters,
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

    def __init__(self, Dataset, generator_param, discriminator_param, hyper_param):

        self.dataset = Dataset()

        self.generator = Model.Generator(
            image_size=generator_param.image_size,
            filters=generator_param.filters,
            block_params=generator_param.block_params,
            data_format=generator_param.data_format
        )

        self.discriminator = Model.Discriminator(
            filters=discriminator_param.filters,
            block_params=discriminator_param.block_params,
            data_format=discriminator_param.data_format
        )

        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.latents = tf.random_normal(shape=[self.batch_size, hyper_param.latent_dimensions])

        self.training = tf.placeholder(dtype=tf.bool, shape=[])
        self.gradient_coefficient = tf.constant(value=hyper_param.gradient_coefficient, dtype=tf.float32)

        self.reals = self.dataset.input()
        self.fakes = self.generator(inputs=self.latents, training=self.training, reuse=False)

        self.real_logits = self.discriminator(inputs=self.reals, training=self.training, reuse=False)
        self.fake_logits = self.discriminator(inputs=self.fakes, training=self.training, reuse=True)

        self.generator_loss = self.wgan_generator_loss()
        self.discriminator_loss = self.wgan_discriminator_loss()
        self.discriminator_loss += self.wgan_gradient_penalty() * self.gradient_coefficient

        self.generator_eval_metric_op = tf.metrics.accuracy(
            labels=tf.ones_like(self.fake_logits),
            predictions=tf.round(self.fake_logits)
        )

        self.discriminator_eval_metric_op = tf.metrics.accuracy(
            labels=tf.concat([tf.ones_like(self.real_logits), tf.zeros_like(self.fake_logits)], axis=0),
            predictions=tf.round(tf.concat([self.real_logits, self.fake_logits], axis=0))
        )

        self.generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        self.discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

        self.generator_global_step = tf.Variable(initial_value=0, trainable=False)
        self.discriminator_global_step = tf.Variable(initial_value=0, trainable=False)

        self.generator_optimizer = tf.train.AdamOptimizer()
        self.discriminator_optimizer = tf.train.AdamOptimizer()

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(self.update_ops):

            self.generator_train_op = self.generator_optimizer.minimize(
                loss=self.generator_loss,
                var_list=self.generator_variables,
                global_step=self.generator_global_step
            )

            self.discriminator_train_op = self.discriminator_optimizer.minimize(
                loss=self.discriminator_loss,
                var_list=self.discriminator_variables,
                global_step=self.discriminator_global_step
            )

    def wgan_generator_loss(self):

        return -tf.reduce_mean(self.fake_logits)

    def wgan_discriminator_loss(self):

        return -tf.reduce_mean(self.real_logits) + tf.reduce_mean(self.fake_logits)

    def wgan_gradient_penalty(self):

        interpolate_coefficients = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], dtype=tf.float32)
        interpolates = self.reals + (self.fakes - self.reals) * interpolate_coefficients
        interpolate_logits = self.discriminator(inputs=interpolates, training=self.training, reuse=True)

        gradients = tf.gradients(interpolate_logits, interpolates)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]) + 0.0001)
        gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0))

        return gradient_penalty

    def initialize(self, model_dir):

        session = tf.get_default_session()

        session.run(tf.local_variables_initializer())

        print("local variables initialized")

        saver = tf.train.Saver()

        checkpoint = tf.train.latest_checkpoint(model_dir)

        if checkpoint:

            saver.restore(session, checkpoint)

            print(checkpoint, "loaded")

        else:

            session.run(tf.global_variables_initializer())

            print("global variables initialized")

        return saver

    def train(self, model_dir, dataset_param, config):

        with tf.Session(config=config) as session:

            saver = self.initialize(model_dir)

            try:

                print("training started")

                start = time.time()

                self.dataset.initialize(
                    filenames=dataset_param.filenames,
                    batch_size=dataset_param.batch_size,
                    num_epochs=dataset_param.num_epochs,
                    buffer_size=dataset_param.buffer_size
                )

                feed_dict = {self.batch_size: dataset_param.batch_size, self.training: True}

                for i in itertools.count():

                    session.run([self.generator_train_op], feed_dict=feed_dict)
                    session.run([self.discriminator_train_op], feed_dict=feed_dict)

                    if i % 100 == 0:

                        generator_global_step, generator_loss = session.run(
                            [self.generator_global_step, self.generator_loss],
                            feed_dict=feed_dict
                        )

                        print("global_step: {}, generator_loss: {:.1f}".format(
                            generator_global_step,
                            generator_loss
                        ))

                        discriminator_global_step, discriminator_loss = session.run(
                            [self.discriminator_global_step, self.discriminator_loss],
                            feed_dict=feed_dict
                        )

                        print("global_step: {}, discriminator_loss: {:.1f}".format(
                            discriminator_global_step,
                            discriminator_loss
                        ))

                        checkpoint = saver.save(
                            sess=session,
                            save_path=os.path.join(model_dir, "model.ckpt"),
                            global_step=generator_global_step
                        )

                        stop = time.time()

                        print("{} saved ({:.1f} sec)".format(checkpoint, stop - start))

                        start = time.time()

                        reals, fakes = session.run([self.reals, self.fakes], feed_dict=feed_dict)

                        images = np.concatenate([reals, fakes], axis=2)

                        for image in images:

                            cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                            cv2.waitKey(1000)

            except tf.errors.OutOfRangeError:

                print("training ended")

    def evaluate(self, model_dir, dataset_param, config):

        with tf.Session(config=config) as session:

            self.initialize(model_dir)

            try:

                print("evaluation started")

                self.dataset.initialize(
                    filenames=dataset_param.filenames,
                    batch_size=dataset_param.batch_size,
                    num_epochs=dataset_param.num_epochs,
                    buffer_size=dataset_param.buffer_size
                )

                feed_dict = {self.batch_size: dataset_param.batch_size, self.training: False}

                for _ in itertools.count():

                    generator_accuracy = session.run([self.generator_eval_metric_op], feed_dict=feed_dict)

                    print("generator_accuracy: {:.1f}".format(generator_accuracy))

                    discriminator_accuracy = session.run([self.discriminator_eval_metric_op], feed_dict=feed_dict)

                    print("discriminator_accuracy: {:.1f}".format(discriminator_accuracy))

            except tf.errors.OutOfRangeError:

                print("evaluation ended")

    def predict(self, model_dir, dataset_param, config):

        with tf.Session(config=config) as session:

            self.initialize(model_dir)

            try:

                print("prediction started")

                self.dataset.initialize(
                    filenames=dataset_param.filenames,
                    batch_size=dataset_param.batch_size,
                    num_epochs=dataset_param.num_epochs,
                    buffer_size=dataset_param.buffer_size
                )

                feed_dict = {self.batch_size: dataset_param.batch_size, self.training: False}

                for _ in itertools.count():

                    reals, fakes = session.run([self.reals, self.fakes], feed_dict=feed_dict)

                    images = np.concatenate([reals, fakes], axis=2)

                    for image in images:

                        cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                        cv2.waitKey(1000)

            except tf.errors.OutOfRangeError:

                print("prediction ended")
