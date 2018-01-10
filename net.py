import tensorflow as tf
import tensorflow.contrib.slim as slim

from collections import OrderedDict


def prep_image(image):
    """
    Image pre-processing operation.
    :param image: tf.placeholder or tf.Tensor, with shape(None, None, 3) and dtype=tf.uint8
    :param input_size: tuple or list, specifying the image height and width
    after resizing.
    :return:
        result: tf.Tensor, with shape(1, None, None, 3), dtype=tf.float32 and pixel value within [0.0, 1.0]
    """
    image = tf.cast(image, dtype=tf.float32)
    images = tf.expand_dims(image, axis=0)
    images = tf.divide(images, 255.0, name='normalize')
    return images


def prep_image_two(images):
    images = tf.cast(images, dtype=tf.float32)
    images = tf.image.random_brightness(images, max_delta=63)
    images = tf.image.random_contrast(images, lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    # images = tf.image.per_image_standardization(images)
    images = tf.map_fn(lambda x: tf.image.per_image_standardization(x), images)
    return images


def prep_image_for_test(image, input_size):
    height, width = input_size
    image = tf.cast(image, dtype=tf.float32)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    images = tf.expand_dims(float_image, axis=0)
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
    def __init__(self, image):
        with tf.name_scope('image_prep'):
            images = prep_image(image)

        with tf.name_scope('FCN'):
            conv1 = conv2d(images, 16, 'conv1', ksize=(5, 5))
            conv2 = conv2d(conv1, 32, 'conv2')
            conv3 = conv2d(conv2, 32, 'conv3')
            conv4 = conv2d(conv3, 64, 'conv4')
            conv5 = conv2d(conv4, 64, 'conv5')

        endpoints = OrderedDict()
        endpoints['images'] = images
        endpoints['conv1'] = conv1
        endpoints['conv2'] = conv2
        endpoints['conv3'] = conv3
        endpoints['conv4'] = conv4
        endpoints['conv5'] = conv5
        self.endpoints = endpoints
