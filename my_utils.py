import os
import time
import json
import pickle
import logging
import numpy as np
import logging.config
import tensorflow as tf

import tensorflow.contrib.slim as slim
from scipy.misc import imresize
from collections import deque


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


def get_config_for_kfold(config, **kwargs):
    copy = config.copy()
    copy.update(kwargs)
    return copy


##################################################################
#                       file utilities                           #
##################################################################
def dump_obj(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def create_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


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
def get_class_weights(labels, class_weights):
    with tf.name_scope('class_weights'):
        labels = tf.cast(labels, dtype=tf.float32)
        w0 = tf.multiply(tf.ones_like(labels), class_weights[0])
        w1 = tf.multiply(tf.ones_like(labels), class_weights[1])
        weights = tf.where(tf.equal(labels, 0), x=w0, y=w1)
    return weights


def metric_summary_op(labels, predictions):
    tp, tp_op = slim.metrics.streaming_true_positives(labels=labels, predictions=predictions)
    tn, tn_op = slim.metrics.streaming_true_negatives(labels=labels, predictions=predictions)
    fp, fp_op = slim.metrics.streaming_false_positives(labels=labels, predictions=predictions)
    fn, fn_op = slim.metrics.streaming_false_negatives(labels=labels, predictions=predictions)
    sensitivity = tf.divide(tp, tp + fn, name='sensitivity')
    specificity = tf.divide(tn, tn + fp, name='specificity')
    accuracy, ac_op = tf.metrics.accuracy(labels=labels, predictions=predictions)
    update_op = tf.group(tp_op, tn_op, fp_op, fn_op, ac_op)
    return accuracy, sensitivity, specificity, update_op


def draw_bbox(images, boxes, name='bounding_box'):
    def box_transform_fn(box):
        top, left, height, width = box[0], box[1], box[2], box[3]
        y_min = top
        x_min = left
        y_max = top + height
        x_max = left + width
        box = tf.stack([y_min, x_min, y_max, x_max])
        return box
    boxes = tf.map_fn(box_transform_fn, boxes)
    boxes = tf.expand_dims(boxes, axis=1)
    return tf.image.draw_bounding_boxes(images, boxes, name=name)


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


def load_model(saver, config):
    sess = tf.get_default_session()
    saver.restore(sess, tf.train.latest_checkpoint(config['train_dir']))


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

    :param x: tf.Tensor, with shape(batch_size, 4)
    :param size: int, expected bounding-box size
    :param name: str,
    :return:
        result: tf.Tensor
    """
    x = tf.multiply(x, size)
    x = tf.floor(x)
    x = tf.cast(x, dtype=tf.int32)
    x = slim.flatten(x, scope=name)
    return x


def bbox_in_range(x):
    """
    Constraint the bounding-box within the image.

    :param x: tf.Tensor, shape like(batch_size, 4) and value within [0.0, 1.0]
    :return:
    """
    def in_range(pos):
        top, left, height, width = pos[0], pos[1], pos[2], pos[3]
        height = tf.cond(top + height < 1.0, true_fn=lambda: height, false_fn=lambda: (1.0 - top))
        width = tf.cond(left + width < 1.0, true_fn=lambda: width, false_fn=lambda: (1.0 - left))
        return tf.stack([top, left, height, width])
    return tf.map_fn(in_range, x)


def crop_bbox(x, pos, limit=None, scope='crop_box'):
    """
    Crop a bounding-box from image(s).

    :param x: tf.Tensor, giving the image(s) to be cropped
    :param pos: tuple or list, with shape(4,) giving the position(top, left, height, width)
    :param limit: tuple or list, with shape(2,) giving the limited bounding of cropped region. If none,
    ignore it and may raise error during runtime.
    :param scope: str, optional parameter.
    :return:
       image: tf.Tensor, the cropped image region
    """
    with tf.name_scope(scope):
        top, left, height, width = pos[0], pos[1], pos[2], pos[3]
        if limit:
            bottom, right = limit
            height = tf.cond(top + height < bottom, true_fn=lambda: height, false_fn=lambda: (bottom - top))
            width = tf.cond(left + width < right, true_fn=lambda: width, false_fn=lambda: (right - left))
        return tf.image.crop_to_bounding_box(x, top, left, height, width)


def crop_bbox_and_resize(x, pos, size,
                         scope='crop_and_resize',
                         limit=None,
                         method=tf.image.ResizeMethod.BILINEAR):
    """
    :param x:
    :param pos:
    :param size:
    :param limit:
    :param method:
    :return:
    """
    with tf.name_scope(scope):
        image = crop_bbox(x, pos, limit=limit, scope=scope)
        image = tf.image.resize_images(image, size=size, method=method)
        return image


def reversed_bbox_transform(x, size, name=None):
    # x = tf.reshape(x, shape=(-1,), name=name)
    x = tf.cast(x, dtype=tf.float32)
    x = tf.divide(x, size, name=name)
    return x


def resize_label(x, size):
    return tf.image.resize_images(x, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


###################################################################
#                        image utilities                          #
###################################################################
def imresize_highest_to(image, h, method='bilinear'):
    height, width = image.shape[:2]
    ratio = h * 1.0 / max(height, width)
    new_h, new_w = int(height * ratio), int(width * ratio)
    return imresize(image, (new_h, new_w), interp=method)


def imresize_smallest_to(image, h, method='bilinear'):
    height, width = image.shape[:2]
    ratio = h * 1.0 / min(height, width)
    new_h, new_w = int(height * ratio), int(width * ratio)
    return imresize(image, (new_h, new_w), interp=method)


def bbox_xy_to_tlwh(x, size):
    """
    Change bounding-box prediction(ymin, xmin, ymax, xmax) back to its size. e.g.(0.1, 0.1, 0.8, 0.8)

    Return:
        result: tuple, four integers giving(top, left, height, width)
    """
    h, w = size
    ymin, xmin, ymax, xmax = x
    top = int(ymin * h)
    left = int(xmin * w)
    height = int((ymax - ymin) * h)
    width = int((xmax - xmin) * w)
    return top, left, height, width


def aug_image(image, label, bbox):
    def randint(low, high):
        if low == high:
            return low
        return np.random.randint(low, high)

    h, w, _ = image.shape
    top, left, height, width = bbox
    new_top = randint(0, top + 1)
    new_left = randint(0, left + 1)
    new_bottom = randint(top + height, h)
    new_right = randint(left + width, w)
    new_image = image[new_top:new_bottom, new_left:new_right]
    new_label = label[new_top:new_bottom, new_left:new_right]
    return new_image, new_label


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
            top = max(0, i - gap)
            break
    for i in reversed(range(h)):
        if any(label[i] == 1):
            bottom = min(h - 1, i + gap)
            break
    for i in range(w):
        if any(label[:, i] == 1):
            left = max(0, i - gap)
            break
    for i in reversed(range(w)):
        if any(label[:, i] == 1):
            right = min(w - 1, i + gap)
            break
    height = bottom - top + 1
    width = right - left + 1
    return np.array([top, left, height, width], dtype=np.int32)


def count_many(predictions, labels):
    """
    Function that return many counter, including TP, TN, FP, FN.
    Labels are assumed to have only two classes, namely 0 and 1.
    :param predictions:
    :param labels:
    :return:
        result: dict
    """
    assert predictions.shape == labels.shape, \
        'shape of predictions%s not equal to labels%s' % (predictions.shape, labels.shape)
    assert (set(np.unique(predictions)) | set(np.unique(labels))) == {0, 1}, \
        '<<ERROR>> classes set not equal {0, 1}'

    correct = (predictions == labels)
    wrong = (predictions != labels)
    true_positives = int(np.sum(correct & (labels == 1)))
    true_negatives = int(np.sum(correct & (labels == 0)))
    false_positives = int(np.sum(wrong & (predictions == 1)))
    false_negatives = int(np.sum(wrong & (predictions == 0)))
    result = {
        'TP': true_positives,
        'TN': true_negatives,
        'FP': false_positives,
        'FN': false_negatives,
    }
    return result


def metric_many_from_counter(result):
    TP, TN, FP, FN = result['TP'], result['TN'], result['FP'], result['FN']
    accuracy = (TP + TN) * 1.0 / (TP + TN + FP + FN)
    sensitivity = TP * 1.0 / (TP + FN)
    specificity = TN * 1.0 / (TN + FP)
    out = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity
    }
    return out


def metric_many_from_predictions(predictions, labels):
    result = count_many(predictions, labels)
    return metric_many_from_counter(result)


def calc_bbox_iou(box_a, box_b):
    """
    Calculate the Intersection over Union metric of bounding-box prediction.

    :param box_a: tuple or list, giving(top, left, height, bottom)
    :param box_b: same as box_a
    :return:
        iou: float,
    """
    def bbox_transform_fn(box):
        top, left, height, width = box
        bottom = top + height
        right = left + width
        return top, left, bottom, right

    def calc_bbox_area(box):
        top, left, bottom, right = box
        if top > bottom or left > right:
            return 0.0
        return (bottom - top + 1) * (right - left + 1)

    box_a = bbox_transform_fn(box_a)
    box_b = bbox_transform_fn(box_b)

    in_top = max(box_a[0], box_b[0])
    in_left = max(box_a[1], box_b[1])
    in_bottom = min(box_a[2], box_b[2])
    in_right = min(box_a[3], box_b[3])
    in_box = (in_top, in_left, in_bottom, in_right)

    in_area = calc_bbox_area(in_box)
    box_a_area = calc_bbox_area(box_a)
    box_b_area = calc_bbox_area(box_b)
    iou = in_area * 1.0 / (box_a_area + box_b_area - in_area)
    return iou


def hole_filling(mask):
    height, width = mask.shape
    idx = 1
    flag = np.zeros_like(mask)
    best_idx = -1
    best_count = -1

    def inrange(pos):
        pi, pj = pos
        return 0 <= pi < height and 0 <= pj < width

    def fill_from(pos, index):
        counter = 1
        queue = deque()
        queue.append(pos)
        DXS = [1, 0, -1, 0]
        DYS = [0, 1, 0, -1]
        while len(queue) != 0:
            tpi, tpj = queue.pop()
            for dx, dy in zip(DXS, DYS):
                npi, npj = tpi + dx, tpj + dy
                if inrange((npi, npj)) and mask[npi, npj] == 1 and flag[npi, npj] == 0:
                    queue.append((npi, npj))
                    flag[npi, npj] = index
                    counter += 1
        return counter

    for i in range(height):
        for j in range(width):
            if mask[i, j] == 0 or flag[i, j] != 0:
                continue
            count = fill_from((i, j), idx)
            if best_count < count:
                best_idx = idx
                best_count = count
            idx += 1
    result = np.zeros_like(mask)
    result[flag == best_idx] = 1
    return result, flag
