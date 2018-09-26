from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import dcgan
import dataset
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="celeba_dcgan_model", help="model directory")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--num_epochs", type=int, default=10, help="number of training epochs")
parser.add_argument("--buffer_size", type=int, default=100000, help="buffer size to shuffle dataset")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument('--predict', action="store_true", help="with prediction")
parser.add_argument('--gpu', type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


class Dataset(dataset.Dataset):

    def parse(self, example):

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

        image = tf.read_file(features["path"])
        image = tf.image.decode_jpeg(image, 3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [256, 256])
        image = utils.scale(image, 0, 1, -1, 1)

        return image


dcgan_model = dcgan.Model(
    Dataset=Dataset,
    generator_param=dcgan.Model.GeneratorParam(
        image_size=[4, 4],
        filters=1024,
        block_params=[dcgan.Model.BlockParam(blocks=1, strides=1)] * 6,
        conv_param=dcgan.Model.ConvParam(kernel_size=3, strides=1),
        data_format="channels_last",
    ),
    discriminator_param=dcgan.Model.DiscriminatorParam(
        filters=32,
        conv_param=dcgan.Model.ConvParam(kernel_size=3, strides=1),
        block_params=[dcgan.Model.BlockParam(blocks=1, strides=1)] * 6,
        data_format="channels_last"
    ),
    hyper_param=dcgan.Model.HyperParam(
        latent_dimension=256,
        gradient_coefficient=5.0
    )
)

if args.train:

    dcgan_model.train(
        model_dir=args.model_dir,
        dataset_param=dcgan.Model.DatasetParam(
            filenames=["data/train.tfrecord"],
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            buffer_size=args.buffer_size,
        ),
        config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list=args.gpu,
                allow_growth=True
            ),
            log_device_placement=False,
            allow_soft_placement=True
        )
    )

if args.predict:

    dcgan_model.evaluate(
        model_dir=args.model_dir,
        dataset_param=dcgan.Model.DatasetParam(
            filenames=["data/test.tfrecord"],
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            buffer_size=args.buffer_size,
        ),
        config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list=args.gpu,
                allow_growth=True
            ),
            log_device_placement=False,
            allow_soft_placement=True
        )
    )

if args.predict:

    dcgan_model.predict(
        model_dir=args.model_dir,
        dataset_param=dcgan.Model.DatasetParam(
            filenames=["data/test.tfrecord"],
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            buffer_size=args.buffer_size,
        ),
        config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list=args.gpu,
                allow_growth=True
            ),
            log_device_placement=False,
            allow_soft_placement=True
        )
    )
