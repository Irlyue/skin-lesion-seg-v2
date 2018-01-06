import tensorflow as tf
import tensorflow.contrib.slim as slim

from collections import OrderedDict


def prep_image(image, input_size):
    """
    Image pre-processing operation.
    :param image: tf.placeholder or tf.Tensor, with shape(None, None, 3) and dtype=tf.uint8
    :param input_size: tuple or list, specifying the image height and width
    after resizing.
    :return:
        result: tf.Tensor, with shape(1, height, width, 3), dtype=tf.float32 and pixel value within [0.0, 1.0]
    """
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.resize_images(image, size=input_size)
    images = tf.expand_dims(image, axis=0)
    images = tf.divide(images, 255.0, name='normalize')
    return images


def conv2d(inputs,
           n_filters,
           scope,
           stride=2,
           ksize=(3, 3),
           activation_fn=tf.nn.relu,
           padding='SAME'):
    return slim.conv2d(inputs,
                       num_outputs=n_filters,
                       scope=scope,
                       stride=stride,
                       activation_fn=activation_fn,
                       kernel_size=ksize,
                       padding=padding)


class FCN:
    def __init__(self, image, input_size):
        with tf.name_scope('image_prep'):
            images = prep_image(image, input_size)

        with tf.name_scope('FCN'):
            conv1 = conv2d(images, 16, 'conv1', ksize=(5, 5))
            conv2 = conv2d(conv1, 32, 'conv2')
            conv3 = conv2d(conv2, 32, 'conv3')
            conv4 = conv2d(conv3, 64, 'conv4')
            conv5 = conv2d(conv4, 128, 'conv5')

        endpoints = OrderedDict()
        endpoints['images'] = images
        endpoints['conv1'] = conv1
        endpoints['conv2'] = conv2
        endpoints['conv3'] = conv3
        endpoints['conv4'] = conv4
        endpoints['conv5'] = conv5
        self.endpoints = endpoints
