import net
import my_utils
import tensorflow as tf
import tensorflow.contrib.slim as slim


logger = my_utils.get_default_logger()


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
            conv6 = slim.conv2d(self.net.endpoints['conv5'],
                                num_outputs=64,
                                kernel_size=(5, 5),
                                stride=1)
            conv6_flatten = tf.reshape(conv6, shape=(1, -1))
            fc6 = slim.fully_connected(conv6_flatten,
                                       num_outputs=4,
                                       activation_fn=tf.nn.sigmoid,
                                       scope='fc6')
            bbox = my_utils.bbox_transform(fc6, input_size[0], name='bbox')

        with tf.name_scope('segmentation'):
            score4 = net.conv2d(self.net.endpoints['conv4'],
                                scope='score4',
                                n_filters=2,
                                stride=1,
                                ksize=(1, 1),
                                activation_fn=None)
            up_score4 = my_utils.up_score_layer(score4,
                                                scope='up_score4',
                                                shape=tf.shape(self.net.endpoints['conv3']),
                                                num_classes=2,
                                                ksize=4,
                                                stride=2)
            score3 = net.conv2d(self.net.endpoints['conv3'],
                                scope='score3',
                                n_filters=2,
                                stride=1,
                                ksize=(1, 1),
                                activation_fn=None)
            fuse_score3 = tf.add(score3, up_score4, name='fuse_score3')
            score = fuse_score3
            up_score = my_utils.up_score_layer(score,
                                               shape=[1, *input_size, 2],
                                               num_classes=2,
                                               ksize=16,
                                               stride=8)
            roi = my_utils.crop_bbox(up_score, bbox, limit=input_size)
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

