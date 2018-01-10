import utils
import model
import inputs
import bbox_model
import evaluation
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.misc import imresize

config = utils.load_config()
logger = utils.get_default_logger()


def test_model():
    image = tf.placeholder(dtype=tf.uint8, name='images', shape=config['input_size'] + [3])
    mm = model.Model(image, config['input_size'])
    logger.info(mm)


def test_huber_loss():
    x = np.linspace(4, -4, 50)
    with tf.Graph().as_default():
        xx = tf.constant(x)
        yy = utils.huber_loss(xx)
        with tf.Session() as sess:
            a, b = sess.run([xx, yy])
    plt.plot(a, b)
    plt.axis('equal')
    plt.show()


def test_bbox_model():
    image = tf.placeholder(dtype=tf.uint8, name='images', shape=(16, 224, 224, 3))
    mm = bbox_model.Model(image, config['input_size'])
    logger.info(mm)


def test_evaluation():
    data = inputs.load_training_data('dermis', config)
    image, label, bbox = data[1]
    bbox_pred, cnn_result, cnn_crf_result = evaluation.inference_one_image_from_prob(image)
    top, left, height, width = bbox_pred

    plt.subplot(231)
    plt.imshow(image)
    plt.subplot(232)
    plt.imshow(label, cmap='gray')
    plt.subplot(233)
    plt.imshow(image)
    plt.gca().add_patch(plt.Rectangle((left, top), width, height, alpha=0.5))
    plt.subplot(234)
    plt.imshow(cnn_result, cmap='gray')
    plt.subplot(235)
    plt.imshow(cnn_crf_result, cmap='gray')
    plt.show()


if __name__ == '__main__':
    test_bbox_model()
    # test_evaluation()
    # test_model()
    # test_huber_loss()
