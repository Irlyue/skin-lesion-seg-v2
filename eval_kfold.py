import sys
import inputs
import my_utils
import evaluation
import tensorflow as tf

logger = my_utils.get_default_logger()


class RestoredModel:
    def __init__(self, ckpt_file):
        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device('/cpu'):
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(ckpt_file, 'rb') as fid:
                    sg = fid.read()
                    od_graph_def.ParseFromString(sg)
                    tf.import_graph_def(od_graph_def, name='')

                self.image_ph = g.get_tensor_by_name('image_tensor:0')
                self.bboxes = g.get_tensor_by_name('detection_boxes:0')
                self.scores = g.get_tensor_by_name('detection_scores:0')
                self.n_bboxes = g.get_tensor_by_name('num_detections:0')

    def inference_box(self, image):
        sess = tf.get_default_session()
        image = image[None] if len(image.shape) == 3 else image
        return sess.run(self.bboxes, feed_dict={self.image_ph: image})[0, 0]


def eval_one_fold(fold, ckpt_path, out_path, ignore_iou=None):
    if ignore_iou:
        logger.warning('Will ignore images with IoU small than %.3f' % ignore_iou)
    config = my_utils.load_config()
    net = RestoredModel(ckpt_path)
    dermquest = inputs.load_raw_data('dermquest', config)
    # train_data = inputs.get_kth_fold(dermquest, fold, config['n_folds'], seed=config['split_seed'])
    test_data = inputs.get_kth_fold(dermquest, fold, config['n_folds'], seed=config['split_seed'], type_='test')
    with net.graph.as_default() as g:
        result = {
            'TP': 0,
            'TN': 0,
            'FP': 0,
            'FN': 0
        }

        def update_dict(target, to_update):
            for key in to_update:
                target[key] += to_update[key]
        with tf.Session(graph=g, config=tf.ConfigProto(device_count={'GPU': 0})):
            counter = 0
            for i, base in enumerate(test_data.listing):
                image, label, bbox_gt = inputs.load_one_example(base, highest_to=800)
                result_i, _ = evaluation.inference_with_restored_model(net, image, label,
                                                                       bbox_gt=bbox_gt,
                                                                       verbose=False,
                                                                       times=3,
                                                                       gt_prob=0.51)
                if ignore_iou and _['IoU'] < ignore_iou:
                    counter += 1
                    print(i, base, '---->')
                    continue
                update_dict(result, result_i)
                result_i.update(my_utils.metric_many_from_counter(result_i))
            result.update(my_utils.metric_many_from_counter(result))
            logger.warning('%d of the images are ignored' % counter)
            logger.info(result)
    my_utils.dump_obj(out_path, result)
    logger.info('Result saved at %s' % out_path)

BASE_DIR = '/home/wenfeng/all-files/skin-lesion-seg-v2/'
CKPT_PATH = BASE_DIR + 'training/train/%d/output_inference_graph.pb/frozen_inference_graph.pb'
OUT_PATH = BASE_DIR + 'training/train/%d/result.pkl'

if __name__ == '__main__':
    sys.argv.append(0.0)
    fold = int(sys.argv[1])
    ignore_iou = float(sys.argv[2])
    eval_one_fold(fold, CKPT_PATH % fold, OUT_PATH % fold, ignore_iou)

