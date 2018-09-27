from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import abc
import os
import time
import cv2


class Model(metaclass=abc.ABCMeta):

    HyperParam = collections.namedtuple("HyperParam", ("latent_size"))

    def __init__(self, dataset, generator, discriminator, hyper_param):

        self.dataset = dataset
        self.generator = generator
        self.discriminator = discriminator

        self.training = tf.placeholder(dtype=tf.bool, shape=[])
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.latent_size = tf.constant(value=hyper_param.latent_size, dtype=tf.int32)
        self.latents = tf.random_normal(shape=[self.batch_size, self.latent_size])

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

        self.generator_loss = self.generator_loss()
        self.discriminator_loss = self.discriminator_loss()

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

    @abstractmethod
    def generator_loss(self):

        pass

    @abstractmethod
    def discriminator_loss(self):

        pass

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

    def initialize_dataset(self, filenames, batch_size, num_epochs, buffer_size):

    def train(self, model_dir, filenames, batch_size, num_epochs, buffer_size, config):

        with tf.Session(config=config) as session:

            saver = self.initialize(model_dir)

            try:

                print("training started")

                start = time.time()

                self.dataset.initialize(
                    filenames=filenames,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    buffer_size=buffer_size
                )

                feed_dict = {self.batch_size: batch_size, self.training: True}

                for i in itertools.count():

                    session.run([self.generator_train_op], feed_dict=feed_dict)
                    session.run([self.discriminator_train_op], feed_dict=feed_dict)

                    if i % 100 == 0:

                        generator_global_step, generator_loss = session.run(
                            [self.generator_global_step, self.generator_loss],
                            feed_dict=feed_dict
                        )

                        print("global_step: {}, generator_loss: {:.2f}".format(
                            generator_global_step,
                            generator_loss
                        ))

                        discriminator_global_step, discriminator_loss = session.run(
                            [self.discriminator_global_step, self.discriminator_loss],
                            feed_dict=feed_dict
                        )

                        print("global_step: {}, discriminator_loss: {:.2f}".format(
                            discriminator_global_step,
                            discriminator_loss
                        ))

                        checkpoint = saver.save(
                            sess=session,
                            save_path=os.path.join(model_dir, "model.ckpt"),
                            global_step=generator_global_step
                        )

                        stop = time.time()

                        print("{} saved ({:.2f} sec)".format(checkpoint, stop - start))

                        start = time.time()

                        reals, fakes = session.run([self.reals, self.fakes], feed_dict=feed_dict)

                        images = np.concatenate([reals, fakes], axis=2)

                        for image in images:

                            cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                            cv2.waitKey(1000)

            except tf.errors.OutOfRangeError:

                print("training ended")

    def evaluate(self, model_dir, filenames, batch_size, num_epochs, buffer_size, config):

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

                    print("generator_accuracy: {:.2f}".format(generator_accuracy))

                    discriminator_accuracy = session.run([self.discriminator_eval_metric_op], feed_dict=feed_dict)

                    print("discriminator_accuracy: {:.2f}".format(discriminator_accuracy))

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
