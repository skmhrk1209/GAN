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

parser = argparse.ArgumentParser()
parser.add_argument("--stp", type=int, default=10000, help="training steps")
parser.add_argument("--siz", type=int, default=16, help="batch size")
parser.add_argument("--dim", type=int, default=100, help="latent dimensions")
args = parser.parse_args()

def generator(inputs, training, reuse=False):

    with tf.variable_scope('generator', reuse=reuse):

        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        dense layer 1
        (-1, 100) -> (-1, 8, 8, 1024)
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        dense1 = tf.layers.dense(
            inputs=inputs,
            units=8 * 8 * 1024,
            activation=tf.nn.relu
        )

        reshape1 = tf.reshape(
            tensor=dense1, 
            shape=(-1, 8, 8, 1024)
        )

        norm1 = tf.layers.batch_normalization(
            inputs=reshape1,
            training=training
        )

        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        deconvolutional layer 2
        (-1, 8, 8, 1024) -> (-1, 16, 16, 512)
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        deconv2 = tf.layers.conv2d_transpose(
            inputs=norm1,
            filters=512,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same", 
            activation=tf.nn.relu
        )

        norm2 = tf.layers.batch_normalization(
            inputs=deconv2,
            training=training
        )

        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        deconvolutional layer 3
        (-1, 16, 16, 512) -> (-1, 32, 32, 256)
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        deconv3 = tf.layers.conv2d_transpose(
            inputs=norm2,
            filters=256,
            kernel_size=(5, 5),
            strides=(2, 2), 
            padding="same", 
            activation=tf.nn.relu
        )

        norm3 = tf.layers.batch_normalization(
            inputs=deconv3,
            training=training
        )

        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        deconvolutional layer 4
        (-1, 32, 32, 256) -> (-1, 64, 64, 128)
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        deconv4 = tf.layers.conv2d_transpose(
            inputs=norm3,
            filters=128,
            kernel_size=(5, 5),
            strides=(2, 2), 
            padding="same", 
            activation=tf.nn.relu
        )

        norm4 = tf.layers.batch_normalization(
            inputs=deconv4,
            training=training
        )

        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        deconvolutional layer 5
        (-1, 64, 64, 128) -> (-1, 128, 128, 3)
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        deconv5 = tf.layers.conv2d_transpose(
            inputs=norm4, 
            filters=3, 
            kernel_size=(5, 5),
            strides=(2, 2), 
            padding="same",
            activation=tf.nn.sigmoid
        )

        return deconv5

def discriminator(inputs, training, reuse=False):

    with tf.variable_scope('discriminator', reuse=reuse):

        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        convolutional layer 1
        (-1, 128, 128, 3) -> (-1, 64, 64, 32)
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        conv1 = tf.layers.conv2d(
            inputs=inputs,
            filters=32,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            activation=tf.nn.leaky_relu
        )

        norm1 = tf.layers.batch_normalization(
            inputs=conv1,
            training=training
        )

        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        convolutional layer 2
        (-1, 64, 64, 32) -> (-1, 32, 32, 64)
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        conv2 = tf.layers.conv2d(
            inputs=norm1,
            filters=64,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            activation=tf.nn.leaky_relu
        )

        norm2 = tf.layers.batch_normalization(
            inputs=conv2,
            training=training
        )

        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        convolutional layer 3
        (-1, 32, 32, 64) -> (-1, 16, 16, 128)
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        conv3 = tf.layers.conv2d(
            inputs=norm2,
            filters=128,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            activation=tf.nn.leaky_relu
        )

        norm3 = tf.layers.batch_normalization(
            inputs=conv3,
            training=training
        )

        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        convolutional layer 4
        (-1, 16, 16, 128) -> (-1, 8, 8, 256)
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        conv4 = tf.layers.conv2d(
            inputs=norm3,
            filters=256,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            activation=tf.nn.leaky_relu
        )

        norm4 = tf.layers.batch_normalization(
            inputs=conv4,
            training=training
        )

        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        logits layer 5
        (-1, 8, 8, 256) -> (-1, 1)
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        reshape5 = tf.reshape(
            tensor=norm4,
            shape=(-1, 8 * 8 * 256)
        )

        logits = tf.layers.dense(
            inputs=reshape5,
            units=1
        )

        return logits

training = tf.placeholder(tf.bool)
latents = tf.placeholder(tf.float32, shape=(None, args.dim))

fakes = generator(latents, training=training)
reals = tf.placeholder(tf.float32, shape=(None, 128, 128, 3))

fake_labels = tf.placeholder(tf.int32, shape=(None))
real_labels = tf.placeholder(tf.int32, shape=(None))
concat_labels = tf.concat([fake_labels, real_labels], axis=0)

fake_logits = discriminator(fakes, training=training)
real_logits = discriminator(reals, training=training, reuse=True)
concat_logits = tf.concat([fake_logits, real_logits], axis=0)

generator_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=fake_labels, logits=fake_logits)
discriminator_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=concat_labels, logits=concat_logits)

generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

    generator_train_op = tf.train.AdamOptimizer().minimize(loss=generator_loss, var_list=generator_variables)
    discriminator_train_op = tf.train.AdamOptimizer().minimize(loss=discriminator_loss, var_list=discriminator_variables)

def listfile(directory, extension):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1] == extension:
                yield os.path.join(root, file)

def scale(inVal, inMin, inMax, outMin, outMax): 
    return outMin + (inVal - inMin) / (inMax - inMin) * (outMax - outMin)

images = np.array([scale(cv2.imread(file).astype(np.float32), 0, 255, 0, 1) for file in listfile("./killmebaby_data", ".png")])

with tf.Session() as session:

    saver = tf.train.Saver()

    session.run(tf.global_variables_initializer())

    for i in range(args.stp):

        noises = np.random.uniform(0.0, 1.0, size=(args.siz, args.dim))
        batch_images = images[np.random.randint(0, images.shape[0], size=args.siz)]
        
        feed_dict = { latents:noises, reals:batch_images, fake_labels:np.ones(args.siz), real_labels:np.ones(args.siz), training:True }
        session.run(generator_train_op, feed_dict=feed_dict)

        feed_dict = { latents:noises, reals:batch_images, fake_labels:np.zeros(args.siz), real_labels:np.ones(args.siz), training:True }
        session.run(discriminator_train_op, feed_dict=feed_dict)

        if i % 100 == 0:

            saver.save(session, "./killmebaby_dcgan_model/model.ckpt", global_step=i)

            for j in range(10):

                noises = np.random.uniform(-1.0, 1.0, size=(1, args.dim))
                images = session.run(fakes, feed_dict={ latents:noises, training:False })
                    
                cv2.imwrite(os.path.join(".", "generated_images", "_".join(["image", str(i).zfill(5), str(j).zfill(3)]) + ".png"), images[0])
