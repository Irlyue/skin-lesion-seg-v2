import os
import sys
import importlib
import tensorflow as tf
import tensorflow.contrib.slim as slim

# dynamically import python module
sys.path.append(os.path.expanduser('~/models/models/research/slim'))
resnet_v1 = importlib.import_module('nets.resnet_v1')
preprocessing_factory = importlib.import_module('preprocessing.preprocessing_factory')


def model_placeholders(config):
    images = tf.placeholder(tf.uint8, shape=(1, *config['input_size'], 3))
    labels = tf.placeholder(tf.int32, shape=(1, *config['input_size']))
    return images, labels


def image_prep(images, is_training):
    images = tf.cast(images, dtype=tf.float32)
    if is_training:
        images = tf.image.random_brightness(images, max_delta=63)
        images = tf.image.random_contrast(images, lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    images = tf.map_fn(lambda x: tf.image.per_image_standardization(x), images)
    return images


class SegModel:
    def __init__(self, images, is_training):
        """

        :param images: tf.Tensor, with shape(1, None, None, 3)
        """
        with tf.name_scope('image_prep'):
            images = image_prep(images, is_training=is_training)

        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            _, endpoints = resnet_v1.resnet_v1_50(images,
                                                       is_training=is_training,
                                                       global_pool=False)

        with tf.name_scope('segmentation'):
            resnet_out = SegModel.get_resnet_block_k(endpoints, 2)
            up_score32 = slim.conv2d_transpose(resnet_out,
                                               num_outputs=2,
                                               kernel_size=(32, 32),
                                               stride=16,
                                               activation_fn=None)

            up_score = up_score32
            mask = tf.argmax(up_score, axis=3, name='mask')
            prob = tf.nn.softmax(up_score, name='prob')
        endpoints['up_score32'] = up_score32
        endpoints['up_score'] = up_score
        endpoints['mask'] = mask
        endpoints['prob'] = prob
        endpoints['images'] = images
        self.endpoints = endpoints

    @staticmethod
    def get_resnet_block_k(endpoints, k):
        fmt = 'resnet_v1_50/block%d'
        return endpoints[fmt % k]

    def __repr__(self):
        myself = ''
        myself += '\n'.join('{:<2} {:<50} {!r}{!r}'.format(i, key, op.dtype, op.shape.as_list())
                            for i, (key, op) in enumerate(self.endpoints.items()))
        return myself


if __name__ == '__main__':
    config = {'input_size': [224, 224]}
    a, b = model_placeholders(config)
    mm = SegModel(a, True)
    print(mm)
