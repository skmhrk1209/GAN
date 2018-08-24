""" implementation of DCGAN in TensorFlow

[1] [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks]
    (https://arxiv.org/pdf/1511.06434.pdf) by Alec Radford, Luke Metz, and Soumith Chintala, Nov 2015.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import sys
import argparse
import functools
import itertools
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="celeba_dcgan_model", help="model directory")
parser.add_argument("--batch", type=int, default=100, help="batch size")
parser.add_argument("--epochs", type=int, default=10, help="training epochs")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument('--predict', action="store_true", help="with prediction")
parser.add_argument('--gpu', type=str, default="0", help="gpu id")
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
            units=2
        )

        return inputs


def parse(example):

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

    image = tf.read_file(features["path"])
    image = tf.image.decode_jpeg(image, 3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, (128, 128))

    return image


filenames = tf.placeholder(tf.string, shape=(None))
training = tf.placeholder(tf.bool, shape=())
buffer_size = tf.placeholder(tf.int64, shape=())
num_epochs = tf.placeholder(tf.int64, shape=())
batch_size = tf.placeholder(tf.int64, shape=())

dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.shuffle(buffer_size)
dataset = dataset.repeat(num_epochs)
dataset = dataset.map(parse)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)

iterator = dataset.make_initializable_iterator()

latents = tf.placeholder(tf.float32, shape=(None, 100))
fakes = generator(latents, training=training)
reals = iterator.get_next()

fake_labels = tf.placeholder(tf.int32, shape=(None))
real_labels = tf.placeholder(tf.int32, shape=(None))
concat_labels = tf.concat([fake_labels, real_labels], axis=0)

fake_logits = discriminator(fakes, training=training)
real_logits = discriminator(reals, training=training, reuse=True)
concat_logits = tf.concat([fake_logits, real_logits], axis=0)

generator_eval_metric_op = tf.metrics.accuracy(
    labels=fake_labels,
    predictions=tf.argmax(fake_logits, axis=1)
)

discriminator_eval_metric_op = tf.metrics.accuracy(
    labels=concat_labels,
    predictions=tf.argmax(concat_logits, axis=1)
)

generator_loss = tf.losses.sparse_softmax_cross_entropy(
    labels=fake_labels,
    logits=fake_logits
)

discriminator_loss = tf.losses.sparse_softmax_cross_entropy(
    labels=concat_labels,
    logits=concat_logits
)

generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

generator_global_step = tf.Variable(0, trainable=False)
discriminator_global_step = tf.Variable(0, trainable=False)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

    generator_train_op = tf.train.AdamOptimizer().minimize(
        loss=generator_loss,
        var_list=generator_variables,
        global_step=generator_global_step
    )

    discriminator_train_op = tf.train.AdamOptimizer().minimize(
        loss=discriminator_loss,
        var_list=discriminator_variables,
        global_step=discriminator_global_step
    )

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list=args.gpu
    ),
    device_count={
        "GPU": 1
    }
)

with tf.Session(config=config) as session:

    saver = tf.train.Saver()

    checkpoint = tf.train.latest_checkpoint(args.model)

    session.run(tf.local_variables_initializer())

    print("local variables initialized")

    if checkpoint:

        saver.restore(session, checkpoint)

        print(checkpoint, "loaded")

    else:

        session.run(tf.global_variables_initializer())

        print("global variables initialized")

    if args.train:

        try:

            print("training started")

            session.run(
                iterator.initializer,
                feed_dict={
                    filenames: ["train.tfrecord"],
                    buffer_size: 180000,
                    num_epochs: args.epochs,
                    batch_size: args.batch
                }
            )

            for i in itertools.count():

                noises = np.random.uniform(
                    low=0.0,
                    high=1.0,
                    size=(args.batch, 100)
                )

                session.run(
                    generator_train_op,
                    feed_dict={
                        latents: noises,
                        fake_labels: np.ones(args.batch),
                        real_labels: np.ones(args.batch),
                        training: True
                    }
                )

                session.run(
                    discriminator_train_op,
                    feed_dict={
                        latents: noises,
                        fake_labels: np.zeros(args.batch),
                        real_labels: np.ones(args.batch),
                        training: True
                    }
                )

                if i % 100 == 0:

                    checkpoint = saver.save(
                        session,
                        os.path.join(args.model, "model.ckpt"),
                        global_step=generator_global_step
                    )

                    print(checkpoint, "saved")

        except tf.errors.OutOfRangeError:

            print("training ended")

    if args.eval:

        try:

            print("evaluating started")

            session.run(
                iterator.initializer,
                feed_dict={
                    filenames: ["eval.tfrecord"],
                    buffer_size: 20000,
                    num_epochs: 1,
                    batch_size: args.batch
                }
            )

            for i in itertools.count():

                noises = np.random.uniform(
                    low=0.0,
                    high=1.0,
                    size=(args.batch, 100)
                )

                generator_accuracy = session.run(
                    generator_eval_metric_op,
                    feed_dict={
                        latents: noises,
                        fake_labels: np.ones(args.batch),
                        real_labels: np.ones(args.batch),
                        training: False
                    }
                )

                print("generator_accuracy", generator_accuracy)

                discriminator_accuracy = session.run(
                    discriminator_eval_metric_op,
                    feed_dict={
                        latents: noises,
                        fake_labels: np.zeros(args.batch),
                        real_labels: np.ones(args.batch),
                        training: False
                    }
                )

                print("discriminator_accuracy", discriminator_accuracy)

        except tf.errors.OutOfRangeError:

            print("evaluating ended")
