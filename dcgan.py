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

    ConvParam = collections.namedtuple("ConvParam", ("kernel_size", "strides"))
    PoolParam = collections.namedtuple("PoolParam", ("pool_size", "strides"))
    BlockParam = collections.namedtuple("BlockParam", ("blocks", "strides"))
    GeneratorParam = collections.namedtuple("GeneratorParam", ("image_size", "filters",
                                                               "block_params", "conv_param", "data_format"))
    DiscriminatorParam = collections.namedtuple("DiscriminatorParam", ("filters", "conv_param", "block_params", "data_format"))
    HyperParam = collections.namedtuple("HyperParam", ("latent_dimension", "gradient_coefficient"))
    DatasetParam = collections.namedtuple("DatasetParam", ("filenames", "batch_size", "num_epochs", "buffer_size"))

    class Generator(object):

        def __init__(self, image_size, filters, block_params, conv_param, data_format):

            self.image_size = image_size
            self.filters = filters
            self.block_params = block_params
            self.conv_param = conv_param
            self.data_format = data_format

        def __call__(self, inputs, training, name="generator", reuse=False):

            with tf.variable_scope(name, reuse=reuse):

                inputs = ops.dense_block(
                    inputs=inputs,
                    units=self.filters * np.prod(self.image_size),
                    normalization=None,
                    activation=tf.nn.leaky_relu,
                    data_format=self.data_format,
                    training=training
                )

                inputs = tf.reshape(
                    tensor=inputs,
                    shape=([-1] + [self.filters] + self.image_size if self.data_format == "channels_first" else
                           [-1] + self.image_size + [self.filters])
                )

                for i, block_param in enumerate(self.block_params):

                    inputs = ops.residual_block(
                        inputs=inputs,
                        filters=self.filters >> i,
                        strides=block_param.strides,
                        normalization=None,
                        activation=tf.nn.leaky_relu,
                        data_format=self.data_format,
                        training=training
                    )

                    for _ in range(1, block_param.blocks):

                        inputs = ops.residual_block(
                            inputs=inputs,
                            filters=self.filters >> i,
                            strides=1,
                            normalization=None,
                            activation=tf.nn.leaky_relu,
                            data_format=self.data_format,
                            training=training
                        )

                    inputs = ops.upsampling2d(
                        inputs=inputs,
                        size=2,
                        data_format=self.data_format
                    )

                inputs = ops.conv2d_block(
                    inputs=inputs,
                    filters=3,
                    kernel_size=self.conv_param.kernel_size,
                    strides=self.conv_param.strides,
                    normalization=None,
                    activation=tf.nn.tanh,
                    data_format=self.data_format,
                    training=training
                )

                return inputs

    class Discriminator(object):

        def __init__(self, filters, conv_param, block_params, data_format):

            self.conv_param = conv_param
            self.filters = filters
            self.block_params = block_params
            self.data_format = data_format

        def __call__(self, inputs, training, name="discriminator", reuse=False):

            with tf.variable_scope(name, reuse=reuse):

                inputs = ops.conv2d_block(
                    inputs=inputs,
                    filters=self.filters,
                    kernel_size=self.conv_param.kernel_size,
                    strides=self.conv_param.strides,
                    normalization=None,
                    activation=tf.nn.leaky_relu,
                    data_format=self.data_format,
                    training=training
                )

                for i, block_param in enumerate(self.block_params):

                    inputs = ops.residual_block(
                        inputs=inputs,
                        filters=self.filters << i,
                        strides=block_param.strides,
                        normalization=None,
                        activation=tf.nn.leaky_relu,
                        data_format=self.data_format,
                        training=training
                    )

                    for _ in range(1, block_param.blocks):

                        inputs = ops.residual_block(
                            inputs=inputs,
                            filters=self.filters << i,
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
            conv_param=generator_param.conv_param,
            data_format=generator_param.data_format
        )

        self.discriminator = Model.Discriminator(
            filters=discriminator_param.filters,
            conv_param=discriminator_param.conv_param,
            block_params=discriminator_param.block_params,
            data_format=discriminator_param.data_format
        )

        self.latents = tf.placeholder(dtype=tf.float32, shape=[None, hyper_param.latent_dimension])
        self.training = tf.placeholder(dtype=tf.bool, shape=[])
        self.gradient_coefficient = tf.constant(value=hyper_param.gradient_coefficient, dtype=tf.float32)

        self.reals = self.dataset.input()

        self.fakes = self.generator(
            inputs=self.latents,
            training=self.training,
            name="generator",
            reuse=False
        )

        self.real_logits = self.discriminator(
            inputs=self.reals,
            training=self.training,
            name="discriminator",
            reuse=False
        )

        self.fake_logits = self.discriminator(
            inputs=self.fakes,
            training=self.training,
            name="discriminator",
            reuse=True
        )

        self.generator_loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=tf.ones_like(self.fake_logits),
            logits=self.fake_logits
        )

        self.discriminator_loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=tf.concat([tf.ones_like(self.real_logits), tf.zeros_like(self.fake_logits)], axis=0),
            logits=tf.concat([self.real_logits, self.fake_logits], axis=0)
        )

        self.gradient_penalty = tf.nn.l2_loss(tf.gradients(ys=self.real_logits, xs=[self.reals])[0])
        self.discriminator_loss += self.gradient_penalty * self.gradient_coefficient

        self.generator_eval_metric_op = tf.metrics.accuracy(
            labels=tf.ones_like(self.fake_logits),
            predictions=tf.round(self.fake_logits)
        )

        self.discriminator_eval_metric_op = tf.metrics.accuracy(
            labels=tf.concat([tf.zeros_like(self.fake_logits), tf.ones_like(self.real_logits)], axis=0),
            predictions=tf.round(tf.concat([self.fake_logits, self.real_logits], axis=0))
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

                for i in itertools.count():

                    latents = np.random.normal(
                        loc=0.0,
                        scale=1.0,
                        size=[dataset_param.batch_size, self.latents.shape[1]]
                    )

                    session.run(
                        [self.generator_train_op],
                        feed_dict={
                            self.latents: latents,
                            self.training: True
                        }
                    )

                    session.run(
                        [self.discriminator_train_op],
                        feed_dict={
                            self.latents: latents,
                            self.training: True
                        }
                    )

                    if i % 100 == 0:

                        generator_global_step, generator_loss = session.run(
                            [self.generator_global_step, self.generator_loss],
                            feed_dict={
                                self.latents: latents,
                                self.training: True
                            }
                        )

                        print("global_step: {}, generator_loss: {:.1f}".format(
                            generator_global_step,
                            generator_loss
                        ))

                        discriminator_global_step, discriminator_loss = session.run(
                            [self.discriminator_global_step, self.discriminator_loss],
                            feed_dict={
                                self.latents: latents,
                                self.training: True
                            }
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

                        reals, fakes = session.run(
                            [self.reals, self.fakes],
                            feed_dict={
                                self.latents: latents,
                                self.training: True
                            }
                        )

                        images = np.concatenate([reals, fakes], axis=2)

                        images = utils.scale(images, -1, 1, 0, 1)

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

                for i in itertools.count():

                    latents = np.random.normal(
                        loc=0.0,
                        scale=1.0,
                        size=[dataset_param.batch_size, self.latents.shape[1]]
                    )

                    generator_accuracy = session.run(
                        [self.generator_eval_metric_op],
                        feed_dict={
                            self.latents: latents,
                            self.training: False
                        }
                    )

                    print("generator_accuracy: {:.1f}".format(generator_accuracy))

                    discriminator_accuracy = session.run(
                        [self.discriminator_eval_metric_op],
                        feed_dict={
                            self.latents: latents,
                            self.training: False
                        }
                    )

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

                for i in itertools.count():

                    latents = np.random.normal(
                        loc=0.0,
                        scale=1.0,
                        size=[dataset_param.batch_size, self.latents.shape[1]]
                    )

                    reals, fakes = session.run(
                        [self.reals, self.fakes],
                        feed_dict={
                            self.latents: latents,
                            self.training: False
                        }
                    )

                    images = np.concatenate([reals, fakes], axis=2)

                    images = utils.scale(images, -1, 1, 0, 1)

                    for image in images:

                        cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                        cv2.waitKey(1000)

            except tf.errors.OutOfRangeError:

                print("prediction ended")
