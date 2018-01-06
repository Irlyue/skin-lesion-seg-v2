import time
import json
import logging
import numpy as np
import logging.config
import tensorflow as tf


class Timer:
    def __init__(self):
        self._tic = time.time()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.eclipsed = time.time() - self._tic


DEFAULT_LOGGER = None


def get_default_logger():
    global DEFAULT_LOGGER
    if DEFAULT_LOGGER is None:
        DEFAULT_LOGGER = logging.getLogger('ALL')
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(name)s %(levelname)s %(message)s'))

        DEFAULT_LOGGER.setLevel(logging.DEBUG)
        DEFAULT_LOGGER.addHandler(handler)
    return DEFAULT_LOGGER


##################################################################
#                       file utilities                           #
##################################################################
def delete_if_exists(path):
    if tf.gfile.Exists(path):
        tf.gfile.DeleteRecursively(path)


def load_config(path=None):
    path = 'config.json' if path is None else path
    with open(path, 'r') as f:
        config = json.load(f)
    return config


###################################################################
#                        tf utilities                             #
###################################################################
def huber_loss(x):
    pass


def up_score_layer(bottom,
                   shape,
                   num_classes,
                   ksize=64,
                   stride=32,
                   scope='up_scope'):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(scope):
        in_features = bottom.get_shape()[3].value

        if shape is None:
            # Compute shape out of Bottom
            in_shape = tf.shape(bottom)

            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, num_classes]
        else:
            new_shape = [shape[0], shape[1], shape[2], num_classes]
        output_shape = tf.stack(new_shape)

        f_shape = [ksize, ksize, num_classes, in_features]
        weights = get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')
    return deconv


def get_deconv_filter(f_shape):
    width = f_shape[0]
    heigh = f_shape[0]
    f = np.ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init,
                           shape=weights.shape)


def bbox_transform(x, size, name=None):
    """
    Transform predicted bounding-box back to the wanted size.

    :param x: tf.Tensor, with shape(4,)
    :param size: int, expected bounding-box size
    :param name: str,
    :return:
        result: tf.Tensor
    """
    x = tf.multiply(x, size)
    x = tf.floor(x)
    x = tf.cast(x, dtype=tf.int32)
    x = tf.reshape(x, shape=(-1,), name=name)
    return x
###################################################################
#                        image utilities                          #
###################################################################


