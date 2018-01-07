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
    def __init__(self, image, input_size, roi_size):
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

        with tf.variable_scope('segmentation'):
            roi_image = utils.crop_bbox_and_resize(image, bbox, roi_size, limit=input_size)
            self.seg_net = net.FCN(roi_image)
            score = net.conv2d(self.seg_net.endpoints['conv4'],
                               scope='conv6',
                               n_filters=128,
                               stride=1,
                               ksize=(3, 3),
                               activation_fn=None)
            up_score = utils.up_score_layer(score,
                                            shape=[1, roi_size[0], roi_size[1], 2],
                                            num_classes=2,
                                            ksize=32,
                                            stride=16)
            lesion_probs = tf.nn.softmax(up_score, name='lesion_probs')
            lesion_mask = tf.expand_dims(tf.argmax(up_score, axis=3), axis=-1, name='lesion_mask')

        endpoints = self.net.endpoints.copy()
        endpoints['fc6'] = fc6
        endpoints['bbox'] = bbox
        for key in self.seg_net.endpoints:
            endpoints['seg/' + key] = self.seg_net.endpoints[key]
        endpoints['up_score'] = up_score
        endpoints['lesion_mask'] = lesion_mask
        endpoints['lesion_probs'] = lesion_probs
        self.endpoints = endpoints
        logger.info('Graph built!')

    def __repr__(self):
        myself = '\n' + '\n'.join('{:>2} {:<15} {!r}{}'.format(i, key, value.dtype, value.shape)
                                  for i, (key, value) in enumerate(self.endpoints.items()))
        return myself

