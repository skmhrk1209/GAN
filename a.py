import tensorflow as tf
import cv2

mnist = tf.contrib.learn.datasets.load_dataset("mnist")

for i, image in enumerate(mnist.train.images):

    cv2.imwrite("mnist/train/mnist/{}.jpg".format(i), image.reshape([28, 28, 1]) * 255)

for i, image in enumerate(mnist.test.images):

    cv2.imwrite("mnist/test/mnist/{}.jpg".format(i), image.reshape([28, 28, 1]) * 255)
