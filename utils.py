import os
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


def calc_training_steps(n_epochs, batch_size, n_examples):
    n_steps_per_epoch = np.ceil(n_examples / batch_size)
    steps = int(n_epochs * n_steps_per_epoch)
    return steps


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


def create_and_delete_if_exists(path):
    delete_if_exists(path)
    os.makedirs(path)

def load_config(path=None):
    path = 'config.json' if path is None else path
    with open(path, 'r') as f:
        config = json.load(f)
    return config


###################################################################
#                        tf utilities                             #
###################################################################
def huber_loss(x, delta=1.0, scope='huber_loss'):
    with tf.name_scope(scope):
        flag = tf.abs(x) < delta
        in_range = 0.5 * tf.square(x)
        out_range = delta * tf.abs(x) - 0.5 * delta**2
        result = tf.where(flag, in_range, out_range)
        return result


def save_model(saver, config):
    sess = tf.get_default_session()
    save_path = os.path.join(config['train_dir'], 'model')
    return saver.save(sess, save_path, global_step=tf.train.get_or_create_global_step())


def add_summary(writer, op, feed_dict):
    sess = tf.get_default_session()
    summary, step = sess.run([op, tf.train.get_or_create_global_step()], feed_dict=feed_dict)
    writer.add_summary(summary, step)


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


def reversed_bbox_transform(x, size, name=None):
    x = tf.reshape(x, shape=(-1,), name=name)
    x = tf.cast(x, dtype=tf.float32)
    x = tf.divide(x, size)
    return x


def resize_label(x, size):
    return tf.image.resize_images(x, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


###################################################################
#                        image utilities                          #
###################################################################
def calc_bbox(label, gap=5):
    """
    Calculate the bounding box given the ground truth label.

    :param label: np.array, with shape(height, width) and dtype=np.int32
    :param gap: int, number of pixels between bounding-box's edge and the object boundary.
    :return:
        result: np.array, with shape(4,) giving (top, left, height, width)
    """
    h, w = label.shape
    top, left, height, width = -1, -1, -1, -1
    for i in range(h):
        if any(label[i] == 1):
            top = i - gap
            break
    for i in reversed(range(h)):
        if any(label[i] == 1):
            bottom = i + gap
            break
    for i in range(w):
        if any(label[:, i] == 1):
            left = i - gap
            break
    for i in reversed(range(w)):
        if any(label[:, i] == 1):
            right = i + gap
            break
    height = bottom - top + 1
    width = right - left + 1
    return np.array([top, left, height, width], dtype=np.int32)
