import net
import utils
import tensorflow as tf


logger = utils.load_config()


class Model:
    def __init__(self, image, input_size):
        logger.info('Building model graph...')
        self.net = net.FCN(image, input_size)
        with tf.name_scope('bbox'):
            bbox = None

        with tf.name_scope('segmentation'):
            lesion_mask = None
            lesion_probs = None

        endpoints = self.net.endpoints.copy()
        endpoints['bbox'] = bbox
        endpoints['lesion_mask'] = lesion_mask
        endpoints['lesion_probs'] = lesion_probs
        self.endpoints = endpoints
        logger.info('Graph built!')

    def __repr__(self):
        pass
