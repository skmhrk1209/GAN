""" implementation of DCGAN in TensorFlow

[1] [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks]
    (https://arxiv.org/pdf/1511.06434.pdf) by Alec Radford, Luke Metz, and Soumith Chintala, Nov 2015.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import os
import sys
import argparse
import functools
import itertools

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="celeba_dcgan_model", help="model directory")
parser.add_argument("--dimension", type=int, default=100, help="latent dimensions")
parser.add_argument("--batch", type=int, default=10, help="batch size")
parser.add_argument("--epochs", type=int, default=100, help="training epochs")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument('--predict', action="store_true", help="with prediction")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def generator(inputs, training, reuse=False):

    with tf.variable_scope("generator", reuse=reuse):

        inputs = tf.layers.dense(
            inputs=inputs,
            units=4 * 4 * 1024
        )

        inputs = tf.reshape(
            tensor=inputs,
            shape=(-1, 4, 4, 1024)
        )

        inputs = tf.layers.batch_normalization(
            inputs=inputs,
            training=training
        )

        inputs = tf.nn.relu(inputs)

        inputs = tf.layers.conv2d_transpose(
            inputs=inputs,
            filters=512,
            kernel_size=5,
            strides=2,
            padding="same"
        )

        inputs = tf.layers.batch_normalization(
            inputs=inputs,
            training=training
        )

        inputs = tf.nn.relu(inputs)

        inputs = tf.layers.conv2d_transpose(
            inputs=inputs,
            filters=256,
            kernel_size=5,
            strides=2,
            padding="same"
        )

        inputs = tf.layers.batch_normalization(
            inputs=inputs,
            training=training
        )

        inputs = tf.nn.relu(inputs)

        inputs = tf.layers.conv2d_transpose(
            inputs=inputs,
            filters=128,
            kernel_size=5,
            strides=2,
            padding="same"
        )

        inputs = tf.layers.batch_normalization(
            inputs=inputs,
            training=training
        )

        inputs = tf.nn.relu(inputs)

        inputs = tf.layers.conv2d_transpose(
            inputs=inputs,
            filters=64,
            kernel_size=5,
            strides=2,
            padding="same"
        )

        inputs = tf.layers.batch_normalization(
            inputs=inputs,
            training=training
        )

        inputs = tf.nn.relu(inputs)

        inputs = tf.layers.conv2d_transpose(
            inputs=inputs,
            filters=3,
            kernel_size=5,
            strides=2,
            padding="same"
        )

        inputs = tf.nn.sigmoid(inputs)

        return inputs


def discriminator(inputs, training, reuse=False):

    with tf.variable_scope("discriminator", reuse=reuse):

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=64,
            kernel_size=5,
            strides=2,
            padding="same"
        )

        inputs = tf.layers.batch_normalization(
            inputs=inputs,
            training=training
        )

        inputs = tf.nn.leaky_relu(inputs)

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=128,
            kernel_size=5,
            strides=2,
            padding="same"
        )

        inputs = tf.layers.batch_normalization(
            inputs=inputs,
            training=training
        )

        inputs = tf.nn.leaky_relu(inputs)

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=256,
            kernel_size=5,
            strides=2,
            padding="same"
        )

        inputs = tf.layers.batch_normalization(
            inputs=inputs,
            training=training
        )

        inputs = tf.nn.leaky_relu(inputs)

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=512,
            kernel_size=5,
            strides=2,
            padding="same"
        )

        inputs = tf.layers.batch_normalization(
            inputs=inputs,
            training=training
        )

        inputs = tf.nn.leaky_relu(inputs)

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=1024,
            kernel_size=5,
            strides=2,
            padding="same"
        )

        inputs = tf.layers.batch_normalization(
            inputs=inputs,
            training=training
        )

        inputs = tf.nn.leaky_relu(inputs)

        inputs = tf.reduce_mean(
            input_tensor=inputs,
            axis=(1, 2)
        )

        inputs = tf.layers.flatten(inputs)

        inputs = tf.layers.dense(
            inputs=inputs,
            units=1
        )

        return inputs


def input_fn(dataset, training, num_examples, num_epochs, batch_size):

    def parse(example, training):

        features = tf.parse_single_example(
            serialized=example,
            features={
                "path": tf.FixedLenFeature(
                    shape=(),
                    dtype=tf.string,
                    default_value=""
                ),
                "label": tf.FixedLenFeature(
                    shape=(),
                    dtype=tf.int64,
                    default_value=0
                )
            }
        )

        path = tf.cast(features["path"], tf.string)
        label = tf.cast(features["label"], tf.int32)

        image = tf.read_file(path)
        image = tf.image.decode_jpeg(image, 3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, (128, 128))

        return image, label

    # how do I get length of dataset ?
    dataset = dataset.shuffle(num_examples)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(functools.partial(parse, training=training))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset.make_one_shot_iterator().get_next()


train_dataset = tf.data.TFRecordDataset("train.tfrecord")

training = tf.placeholder(tf.bool)
latents = tf.placeholder(tf.float32, shape=(None, args.dimension))

fakes = generator(latents, training=training)
reals, _ = input_fn(dataset=train_dataset, training=training, num_examples=686,
                    num_epochs=args.epochs, batch_size=args.batch)

fake_labels = tf.placeholder(tf.int32, shape=(None))
real_labels = tf.placeholder(tf.int32, shape=(None))
concat_labels = tf.concat([fake_labels, real_labels], axis=0)

fake_logits = discriminator(fakes, training=training)
real_logits = discriminator(reals, training=training, reuse=True)
concat_logits = tf.concat([fake_logits, real_logits], axis=0)

generator_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=fake_labels, logits=fake_logits)
discriminator_loss = tf.losses.sigmoid_cross_entropy(
    multi_class_labels=concat_labels, logits=concat_logits)

generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

generator_global_step = tf.Variable(0, trainable=False)
discriminator_global_step = tf.Variable(0, trainable=False)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

    generator_train_op = tf.train.AdamOptimizer().minimize(
        loss=generator_loss, var_list=generator_variables, global_step=generator_global_step)
    discriminator_train_op = tf.train.AdamOptimizer().minimize(
        loss=discriminator_loss, var_list=discriminator_variables, global_step=discriminator_global_step)

with tf.Session() as session:

    saver = tf.train.Saver()

    checkpoint = tf.train.latest_checkpoint(args.model)

    if checkpoint:

        saver.restore(session, checkpoint)

        print(checkpoint, "loaded")

    else:

        session.run(tf.global_variables_initializer())

        print("global variables initialized")

    try:

        for i in itertools.count():

            noises = np.random.uniform(0.0, 1.0, size=(args.batch, args.dimension))

            feed_dict = {latents: noises, fake_labels: np.ones(
                args.batch), real_labels: np.ones(args.batch), training: True}
            session.run(generator_train_op, feed_dict=feed_dict)

            feed_dict = {latents: noises, fake_labels: np.zeros(
                args.batch), real_labels: np.ones(args.batch), training: True}
            session.run(discriminator_train_op, feed_dict=feed_dict)

            if i % 100 == 0:

                checkpoint = saver.save(session, os.path.join(args.model, "model.ckpt"), global_step=generator_global_step)

                print(checkpoint, "saved")

    except tf.errors.OutOfRangeError:

        pass
