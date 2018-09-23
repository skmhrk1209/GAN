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
import dcgan

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="celeba_dcgan_model", help="model directory")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--num_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument('--predict', action="store_true", help="with prediction")
parser.add_argument('--gpu', type=str, default="0", help="gpu id")
parser.add_argument('--data_format', type=str, default="channels_last", help="data_format")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def scale(input, input_min, input_max, output_min, output_max):

    return output_min + (input - input_min) / (input_max - input_min) * (output_max - output_min)


def parse_fn(example):

    features = tf.parse_single_example(
        serialized=example,
        features={
            "path": tf.FixedLenFeature(
                shape=[],
                dtype=tf.string,
                default_value=""
            ),
            "label": tf.FixedLenFeature(
                shape=[],
                dtype=tf.int64,
                default_value=0
            )
        }
    )

    return features["path"]


def preprocess(path, data_format, training):

    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, 3)
    image = tf.image.resize_images(image, [256, 256])
    image = tf.transpose(image, [2, 0, 1]) if data_format == "channels_first" else image
    image = scale(image, 0., 255., -1., 1.)

    return image


filenames = tf.placeholder(tf.string, shape=[None])
batch_size = tf.placeholder(tf.int64, shape=[])
num_epochs = tf.placeholder(tf.int64, shape=[])
buffer_size = tf.placeholder(tf.int64, shape=[])
data_format = tf.placeholder(tf.string, shape=[])
training = tf.placeholder(tf.bool, shape=[])

dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.shuffle(buffer_size)
dataset = dataset.repeat(num_epochs)
dataset = dataset.map(parse_fn)
dataset = dataset.map(
    functools.partial(
        preprocess,
        data_format=args.data_format,
        training=training
    )
)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)

iterator = dataset.make_initializable_iterator()

generator = dcgan.Model.Generator(
    image_size=[256, 256],
    filters=1024,
    block_params=[
        dcgan.Model.BlockParam(
            blocks=1,
            strides=1
        )
    ] * 6,
    bottleneck=False,
    version=2,
    final_conv_param=dcgan.Model.ConvParam(
        kernel_size=3,
        strides=1
    )
)

discriminator = dcgan.Model.Discriminator(
    filters=32,
    initial_conv_param=dcgan.Model.ConvParam(
        kernel_size=3,
        strides=1
    ),
    block_params=[
        dcgan.Model.BlockParam(
            blocks=1,
            strides=1
        )
    ] * 6,
    bottleneck=False,
    version=2
)

latents = tf.placeholder(tf.float32, shape=[None, 256])
fakes = generator(latents, data_format=args.data_format, training=training)
reals = iterator.get_next()

fake_labels = tf.placeholder(tf.int32, shape=[None, 1])
real_labels = tf.placeholder(tf.int32, shape=[None, 1])
concat_labels = tf.concat([fake_labels, real_labels], axis=0)

fake_logits = discriminator(fakes, data_format=args.data_format, training=training)
real_logits = discriminator(reals, data_format=args.data_format, training=training, reuse=True)
concat_logits = tf.concat([fake_logits, real_logits], axis=0)

generator_eval_metric_op = tf.metrics.accuracy(
    labels=fake_labels,
    predictions=tf.map_fn(
        fn=lambda logit: tf.cast(tf.round(logit), tf.int32),
        elems=fake_logits
    )
)

discriminator_eval_metric_op = tf.metrics.accuracy(
    labels=concat_labels,
    predictions=tf.map_fn(
        fn=lambda logit: tf.cast(tf.round(logit), tf.int32),
        elems=concat_logits
    )
)

generator_loss = tf.losses.sigmoid_cross_entropy(
    multi_class_labels=fake_labels,
    logits=fake_logits
)

discriminator_loss = tf.losses.sigmoid_cross_entropy(
    multi_class_labels=concat_labels,
    logits=concat_logits
)

gradient = tf.gradients(ys=real_logits, xs=[reals])
print(len(gradient))
gradient_penalty = tf.nn.l2_loss(gradient) * 10.0

discriminator_loss += gradient_penalty

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
        visible_device_list=args.gpu,
        allow_growth=True
    ),
    log_device_placement=False,
    allow_soft_placement=True
)

with tf.Session(config=config) as session:

    saver = tf.train.Saver()

    checkpoint = tf.train.latest_checkpoint(args.model_dir)

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
                    filenames: ["data/train.tfrecord"],
                    batch_size: args.batch_size,
                    num_epochs: args.num_epochs,
                    buffer_size: 180000
                }
            )

            for i in itertools.count():

                noises = np.random.normal(loc=0.0, scale=1.0, size=[args.batch_size, 256])

                session.run(
                    generator_train_op,
                    feed_dict={
                        latents: noises,
                        fake_labels: np.ones([args.batch_size, 1]),
                        real_labels: np.ones([args.batch_size, 1]),
                        training: True
                    }
                )

                session.run(
                    discriminator_train_op,
                    feed_dict={
                        latents: noises,
                        fake_labels: np.zeros([args.batch_size, 1]),
                        real_labels: np.ones([args.batch_size, 1]),
                        training: True
                    }
                )

                if i % 100 == 0:

                    checkpoint = saver.save(
                        session,
                        os.path.join(args.model_dir, "model.ckpt"),
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
                    filenames: ["data/test.tfrecord"],
                    batch_size: args.batch_size,
                    num_epochs: 1,
                    buffer_size: 20000
                }
            )

            for i in itertools.count():

                noises = np.random.normal(loc=0.0, scale=1.0, size=[args.batch_size, 256])

                generator_accuracy = session.run(
                    generator_eval_metric_op,
                    feed_dict={
                        latents: noises,
                        fake_labels: np.ones([args.batch_size, 1]),
                        real_labels: np.ones([args.batch_size, 1]),
                        training: False
                    }
                )

                print("generator_accuracy", generator_accuracy)

                discriminator_accuracy = session.run(
                    discriminator_eval_metric_op,
                    feed_dict={
                        latents: noises,
                        fake_labels: np.zeros([args.batch_size, 1]),
                        real_labels: np.ones([args.batch_size, 1]),
                        training: False
                    }
                )

                print("discriminator_accuracy", discriminator_accuracy)

        except tf.errors.OutOfRangeError:

            print("evaluating ended")

    if args.predict:

        for i in itertools.count():

            noises = np.random.normal(loc=0.0, scale=1.0, size=[1, 256])

            images = session.run(
                fakes,
                feed_dict={
                    latents: noises,
                    training: False
                }
            )

            cv2.imshow("image", scale(images[0], -1., 1., 0., 255.))

            if cv2.waitKey(1000) == ord("q"):

                break
