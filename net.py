import tensorflow as tf

from collections import OrderedDict


def prep_image(image, input_size):
    """
    Image pre-processing operation.
    :param image: tf.placeholder or tf.Tensor, with shape(None, None, 3)
    :param input_size: tuple or list, specifying the image height and width
    after resizing.
    :return:
        result: tf.Tensor, with shape(1, height, width, 3) and zero-mean/unit-variance.
    """
    pass


class FCN:
    def __init__(self, image, input_size):
        with tf.name_scope('image_prep'):
            images = prep_image(image, input_size)

        with tf.name_scope('FCN'):
            conv1 = None
            conv2 = None
            conv3 = None
            conv4 = None
            conv5 = None

        endpoints = OrderedDict()
        endpoints['conv1'] = conv1
        endpoints['conv2'] = conv2
        endpoints['conv3'] = conv3
        endpoints['conv4'] = conv4
        endpoints['conv5'] = conv5
        self.endpoints = endpoints
