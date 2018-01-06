import utils
import model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    test_model()
    # test_huber_loss()
