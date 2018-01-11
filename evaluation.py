import crf
import model
import my_utils
import bbox_model
import numpy as np
import tensorflow as tf

from crf import crf_post_process
from net import prep_image_for_test


logger = my_utils.get_default_logger()


class EvalModelAbstract:
    def __init__(self):
        pass

    def load_self(self, config):
        global_step = tf.train.get_or_create_global_step()
        self.session = tf.Session().__enter__()
        saver = tf.train.Saver()
        my_utils.load_model(saver, config)
        logger.info('Model-%i restored successfully!' % self.session.run(global_step,))

    def inference(self, image, ops):
        if type(ops[0]) == str:
            ops = [self.model.endpoints[key] for key in ops]
        feed_dict = self._build_feed_dict(image)
        return self.session.run(ops, feed_dict)

    def _build_feed_dict(self, image):
        return {self.image: image}


class EvalModel(EvalModelAbstract):
    def __init__(self, config):
        super().__init__()
        self.image, _, _ = model.model_placeholder(config)
        self.model = model.Model(self.image, config['input_size'])
        self.load_self(config)


class EvalBboxModel(EvalModelAbstract):
    def __init__(self, config):
        super().__init__()
        self.image = tf.placeholder(tf.uint8, shape=(1, *config['input_size'], 3))
        self.model = bbox_model.Model(self.image, config['input_size'],
                                      prep_func=prep_image_for_test)
        self.load_self(config)


def inference_one_image_from_prob(image):
    config = my_utils.load_config()
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        image_ph, _, _ = model.model_placeholder(config)
        mm = model.Model(image_ph, config['input_size'])

        def build_feed_dict(x):
            return {image_ph: x}

        saver = tf.train.Saver()
        with tf.Session() as sess:
            my_utils.load_model(saver, config)
            logger.info('Model-%i restored successfully!' % sess.run(global_step,))
            ops = [mm.endpoints['bbox'], mm.endpoints['lesion_mask'], mm.endpoints['lesion_prob']]
            bbox, lesion_mask, lesion_prob = sess.run(ops, build_feed_dict(image))
        cnn_result = get_cnn_result(image.shape[:-1], bbox, lesion_mask)
        cnn_crf_result = get_cnn_crf_result(image, bbox, lesion_prob)
    return bbox, cnn_result, cnn_crf_result


def get_cnn_result(shape, bbox, mask):
    result = np.zeros(shape)
    mask = np.squeeze(mask)
    top, left, height, width = bbox
    assert mask.shape == (height, width), "mask shape%s not equal to bbox%s" % (mask.shape, bbox)
    result[top:top+height, left:left+width] = mask
    return result


def get_cnn_crf_result(image, bbox, lesion_prob):
    h, w, _ = image.shape
    p0 = np.ones((h, w))
    p1 = np.zeros((h, w))
    top, left, height, width = bbox
    lesion_prob = np.squeeze(lesion_prob)
    assert (height, width) == lesion_prob.shape[:-1], "prob shape%s not equal to bbox%s" % (lesion_prob.shape[:-1], bbox)
    p0[top:top+height, left:left+width] = lesion_prob[:, :, 0]
    p1[top:top+height, left:left+width] = lesion_prob[:, :, 1]
    prob = np.stack([p0, p1], axis=2)
    result = crf_post_process(image, prob)
    return result


def inference_image(net, feed_dict):
    sess = tf.get_default_session()
    bbox, = sess.run([net.endpoints['bbox']], feed_dict)
    return bbox,


def evaluate_one_model(mm, data, config):
    gt_prob = config['gt_prob']
    result = {
        'TP': 0,
        'TN': 0,
        'FP': 0,
        'FN': 0
    }
    n_examples_for_test = len(data)
    logger.info('Evaluating %d examples in total...' % n_examples_for_test)

    def update_dict(target, to_update):
        for key in to_update:
            target[key] += to_update[key]

    def inference_bbox(mm, image_):
        return mm.inference(image_, ['bbox'])[0]

    iou = []
    for i, (image, label, bbox_gt) in enumerate(data):
        bbox_pred = inference_bbox(mm, image[None])[0]
        bbox_crf_result_i = crf.crf_from_bbox(image, bbox_pred, gt_prob)
        result_i = my_utils.count_many(bbox_crf_result_i, label)
        update_dict(result, result_i)

        iou_i = my_utils.calc_bbox_iou(bbox_pred, bbox_gt)
        iou.append(iou_i)

    result.update(my_utils.metric_many_from_counter(result))
    result['mIoU'] = np.mean(iou)
    return result
