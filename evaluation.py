import model
import utils
import tensorflow as tf


logger = utils.get_default_logger()


def inference_one_image(image):
    config = utils.load_config()
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        image_ph, _, _ = model.model_placeholder(config)
        mm = model.Model(image_ph, config['input_size'])

        def build_feed_dict(image):
            return {image_ph: image}

        saver = tf.train.Saver()
        with tf.Session() as sess:
            utils.load_model(saver, config)
            logger.info('Model-%i restored successfully!' % sess.run(global_step,))
            bbox, = sess.run([mm.endpoints['bbox']], build_feed_dict(image))
    return bbox


def inference_image(net, feed_dict):
    sess = tf.get_default_session()
    bbox, = sess.run([net.endpoints['bbox']], feed_dict)
    return bbox,

