import os
import crf
import json
import utils
import inputs
import evaluation
import tensorflow as tf


logger = utils.get_default_logger()


def kfold_evaluation():
    logger.info('K-fold evaluation process...')
    config = utils.load_config()
    dermis = inputs.load_training_data('dermis', config)
    dermquest = inputs.load_training_data('dermquest', config)
    n_folds = config['n_folds']

    for i in range(n_folds):
        test_data = inputs.get_kth_fold(dermquest, i, n_folds,
                                        type_='test',
                                        seed=config['split_seed'])

        kfold_config = utils.get_config_for_kfold(config,
                                                  train_dir=os.path.join(config['train_dir'], str(i)))
        logger.info('Evaluating for %i-th fold data...' % i)
        result = evaluate_one_fold(test_data, kfold_config)
        logger.info('Result for %d-th: \n%s' % (i, json.dumps(result, indent=2)))
        logger.info('************************************\n\n')
    logger.info('Done evaluation')


def evaluate_one_fold(data, config):
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

    with tf.Graph().as_default():
        model = evaluation.EvalModel(config)

        for i, (image, label, _) in enumerate(data):
            bbox_pred = inference_bbox(model, image)
            bbox_crf_result_i = crf.crf_from_bbox(image, bbox_pred, gt_prob)
            result_i = utils.count_many(bbox_crf_result_i, label)
            update_dict(result, result_i)
        result.update(utils.metric_many_from_counter(result))
    return result


if __name__ == '__main__':
    kfold_evaluation()
