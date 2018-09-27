from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import gan_gp
import resnet
import dataset
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="celeba_gan_gp_model", help="model directory")
parser.add_argument("--batch_size", type=int, default=10, help="batch size")
parser.add_argument("--num_epochs", type=int, default=10, help="number of training epochs")
parser.add_argument("--buffer_size", type=int, default=100000, help="buffer size to shuffle dataset")
parser.add_argument('--data_format', type=str, choices=["channels_first", "channels_last"], default="channels_last", help="data_format")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument('--predict', action="store_true", help="with prediction")
parser.add_argument('--gpu', type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


class Dataset(dataset.Dataset):

    def __init__(self, data_format):

        self.data_format = data_format

        super().__init__()

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
        image = tf.image.resize_image_with_crop_or_pad(image, 128, 128)

        if self.data_format == "channels_first":

            image = tf.transpose(image, [2, 0, 1])

        return image


gan_gp_model = gan_gp.Model(
    dataset=Dataset(args.data_format),
    generator=resnet.Generator(
        image_size=[128, 128],
        filters=512,
        residual_params=[
            resnet.Generator.ResidualParam(filters=512, blocks=1),
            resnet.Generator.ResidualParam(filters=256, blocks=1),
            resnet.Generator.ResidualParam(filters=256, blocks=1),
            resnet.Generator.ResidualParam(filters=128, blocks=1),
            resnet.Generator.ResidualParam(filters=64, blocks=1)
        ],
        data_format=args.data_format,
    ),
    discriminator=resnet.Discriminator(
        filters=64,
        residual_params=[
            resnet.Generator.ResidualParam(filters=64, blocks=1),
            resnet.Generator.ResidualParam(filters=128, blocks=1),
            resnet.Generator.ResidualParam(filters=256, blocks=1),
            resnet.Generator.ResidualParam(filters=256, blocks=1),
            resnet.Generator.ResidualParam(filters=512, blocks=1),
            resnet.Generator.ResidualParam(filters=512, blocks=1)
        ],
        data_format=args.data_format
    ),
    hyper_param=gan_gp.Model.HyperParam(
        latent_size=128,
        gradient_coefficient=10.0
    )
)

if args.train:

    gan_gp_model.train(
        model_dir=args.model_dir,
        filenames=["data/train.tfrecord"],
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        buffer_size=args.buffer_size,
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

    gan_gp_model.evaluate(
        model_dir=args.model_dir,
        filenames=["data/test.tfrecord"],
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        buffer_size=args.buffer_size,
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

    gan_gp_model.predict(
        model_dir=args.model_dir,
        filenames=["data/test.tfrecord"],
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        buffer_size=args.buffer_size,
        config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list=args.gpu,
                allow_growth=True
            ),
            log_device_placement=False,
            allow_soft_placement=True
        )
    )
