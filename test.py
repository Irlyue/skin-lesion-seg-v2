import utils
import model
import inputs
import evaluation
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.misc import imresize

config = utils.load_config()
logger = utils.get_default_logger()


def test_model():
    image = tf.placeholder(dtype=tf.uint8, name='images', shape=config['input_size'] + [3])
    mm = model.Model(image, config['input_size'], config['roi_size'])
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


def test_evaluation():
    data = inputs.load_raw_data('dermquest', config)
    image, label, bbox = data[55]
    image_in = imresize(image, config['input_size'])
    bbox_pred = evaluation.inference_one_image(image_in)
    top, left, height, width = bbox_pred

    plt.subplot(221)
    plt.imshow(image_in)
    plt.subplot(222)
    plt.imshow(label, cmap='gray')
    plt.subplot(223)
    plt.imshow(image_in)
    plt.gca().add_patch(plt.Rectangle((left, top), width, height, alpha=0.5))
    plt.show()


if __name__ == '__main__':
    # test_evaluation()
    test_model()
    # test_huber_loss()
