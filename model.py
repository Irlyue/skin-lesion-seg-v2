import net
import utils
import tensorflow as tf
import tensorflow.contrib.slim as slim


logger = utils.get_default_logger()


class Model:
    def __init__(self, image, input_size):
        """
        :param image: tf.placeholder or tf.Tensor, one single image with shape(None, None, 3) and dtype=tf.uint8
        :param input_size: list or tuple,
        """
        self.input_size = input_size
        logger.info('Building model graph...')
        self.net = net.FCN(image, input_size)
        with tf.name_scope('bbox'):
            conv5_flatten = tf.reshape(self.net.endpoints['conv5'], shape=(1, -1))
            fc6 = slim.fully_connected(conv5_flatten,
                                       num_outputs=4,
                                       activation_fn=tf.nn.sigmoid,
                                       scope='fc6')
            bbox = utils.bbox_transform(fc6, input_size[0], name='bbox')

        with tf.name_scope('segmentation'):
            conv5_bbox = utils.bbox_transform(fc6, input_size[0] // 32)
            x, y, h, w = conv5_bbox[0], conv5_bbox[1], conv5_bbox[2], conv5_bbox[3]
            roi = tf.image.crop_to_bounding_box(self.net.endpoints['conv5'], x, y, h, w)
            score = slim.conv2d(roi,
                                num_outputs=2,
                                kernel_size=(3, 3),
                                stride=1,
                                padding='SAME',
                                activation_fn=None,
                                scope='score')
            up_shape = [1, bbox[2], bbox[3], 2]
            up_score = utils.up_score_layer(score,
                                            up_shape,
                                            num_classes=2)
            lesion_probs = tf.nn.softmax(up_score, name='lesion_probs')
            lesion_mask = tf.argmax(up_score, axis=3, name='lesion_mask')

        endpoints = self.net.endpoints.copy()
        endpoints['fc6'] = fc6
        endpoints['bbox'] = bbox
        endpoints['lesion_mask'] = lesion_mask
        endpoints['lesion_probs'] = lesion_probs
        self.endpoints = endpoints
        logger.info('Graph built!')

    def __repr__(self):
        myself = '\n' + '\n'.join('{:>2} {:<10} {!r}{}'.format(i, key, value.dtype, value.shape)
                                  for i, (key, value) in enumerate(self.endpoints.items()))
        return myself

