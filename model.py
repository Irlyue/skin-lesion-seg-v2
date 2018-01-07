import net
import utils
import tensorflow as tf
import tensorflow.contrib.slim as slim


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
        self.net = net.FCN(image)
        with tf.name_scope('bbox'):
            conv5_flatten = tf.reshape(self.net.endpoints['conv5'], shape=(1, -1))
            fc6 = slim.fully_connected(conv5_flatten,
                                       num_outputs=4,
                                       activation_fn=tf.nn.sigmoid,
                                       scope='fc6')
            bbox = utils.bbox_transform(fc6, input_size[0], name='bbox')

        with tf.name_scope('segmentation'):
            score = net.conv2d(self.net.endpoints['conv5'],
                               scope='conv6',
                               n_filters=128,
                               stride=1,
                               ksize=(3, 3),
                               activation_fn=None)
            up_score = utils.up_score_layer(score,
                                            shape=[1, *input_size, 2],
                                            num_classes=2,
                                            ksize=64,
                                            stride=32)
            roi = utils.crop_bbox(up_score, bbox, limit=input_size)
            lesion_prob = tf.nn.softmax(roi, name='lesion_prob')
            lesion_mask = tf.expand_dims(tf.argmax(roi, axis=3), axis=-1, name='lesion_mask')

        endpoints = self.net.endpoints.copy()
        endpoints['fc6'] = fc6
        endpoints['bbox'] = bbox
        endpoints['score'] = score
        endpoints['up_score'] = up_score
        endpoints['roi'] = roi
        endpoints['lesion_mask'] = lesion_mask
        endpoints['lesion_prob'] = lesion_prob
        self.endpoints = endpoints
        logger.info('Graph built!')

    def __repr__(self):
        myself = '\n' + '\n'.join('{:>2} {:<15} {!r}{}'.format(i, key, value.dtype, value.shape)
                                  for i, (key, value) in enumerate(self.endpoints.items()))
        return myself

