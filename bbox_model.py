import net
import utils
import tensorflow as tf
import tensorflow.contrib.slim as slim

from collections import OrderedDict


logger = utils.get_default_logger()


def model_placeholder(config):
    height, width = config['input_size']
    image = tf.placeholder(tf.uint8, name='image_ph', shape=(height, width, 3))
    label = tf.placeholder(tf.int32, name='label_ph', shape=(height, width))
    bbox = tf.placeholder(tf.int32, name='bbox_ph', shape=(4,))
    return image, label, bbox


class Model:
    def __init__(self, image, input_size):
        """
        :param image: tf.placeholder or tf.Tensor, one single image with shape(None, None, 3) and dtype=tf.uint8
        :param input_size: list or tuple,
        """
        self.input_size = input_size
        logger.info('Building model graph...')
        with tf.name_scope('image_prep'):
            images = net.prep_image(image)

        with tf.name_scope('FCN'):
            conv1 = net.conv2d(images, 16, 'conv1', ksize=(5, 5), stride=(4, 4))
            conv2 = net.conv2d(conv1, 32, 'conv2', ksize=(5, 5))
            conv3 = net.conv2d(conv2, 32, 'conv3')
            conv4 = net.conv2d(conv3, 64, 'conv4')
            conv5 = net.conv2d(conv4, 64, 'conv5')
            conv5_flatten = tf.reshape(conv5, shape=(1, -1))
        with tf.name_scope('fully_connected'):
            fc6 = slim.fully_connected(conv5_flatten,
                                       num_outputs=128,
                                       scope='fc6')
            fc7 = slim.fully_connected(fc6,
                                       num_outputs=4,
                                       activation_fn=tf.nn.sigmoid,
                                       scope='fc7')
            bbox = utils.bbox_transform(fc7, input_size[0], name='bbox')

        endpoints = OrderedDict()
        endpoints['images'] = images
        endpoints['conv1'] = conv1
        endpoints['conv2'] = conv2
        endpoints['conv3'] = conv3
        endpoints['conv4'] = conv4
        endpoints['conv5'] = conv5
        endpoints['fc6'] = fc6
        endpoints['fc7'] = fc7
        endpoints['bbox'] = bbox
        self.endpoints = endpoints

    def __repr__(self):
        myself = '\n' + '\n'.join('{:>2} {:<15} {!r}{}'.format(i, key, value.dtype, value.shape)
                                  for i, (key, value) in enumerate(self.endpoints.items()))
        return myself
