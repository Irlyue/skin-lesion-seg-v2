import os
import crf
import json
import my_utils
import inputs
import evaluation
import tensorflow as tf


logger = my_utils.get_default_logger()


def kfold_evaluation():
    cnns, crfs, crf_labels = [], [], []
    logger.info('K-fold evaluation process...')
    config = my_utils.load_config()
    dermquest = inputs.load_training_data('dermquest', config)
    n_folds = config['n_folds']

    for i in range(n_folds):
        test_data = inputs.get_kth_fold(dermquest, i, n_folds,
                                        type_='test',
                                        seed=config['split_seed'])

        kfold_config = my_utils.get_config_for_kfold(config,
                                                     train_dir=os.path.join(config['train_dir'], str(i)))
        logger.info('Evaluating for %i-th fold data...' % i)
        mm = evaluation.SegRestoredModel(tf.train.latest_checkpoint(config['train_dir']))
        cnn_result, crf_result, crf_label_result = test_one_model(mm, test_data, kfold_config)
        cnns.append(cnn_result)
        crfs.append(crf_result)
        crf_labels.append(crf_label_result)
        logger.info('************************************\n\n')
    logger.info('Done evaluation')
    return cnns, crfs, crf_labels


def test_one_model(model, listing, config, use_bbox=False, gt_prob=0.6):
    cnn_result = {
        'TP': 0,
        'TN': 0,
        'FP': 0,
        'FN': 0
    }

    def update_dict(target, to_update):
        for key in to_update:
            target[key] += to_update[key]

    crf_result = cnn_result.copy()
    crf_label_result = cnn_result.copy()
    for base in listing:
        image, label, bbox = inputs.load_one_example(base, size=config['input_size'])
        cnn_out, prob = model.inference(image, ['mask', 'prob'])

        cnn_out, prob = cnn_out[0], prob[0]

        unary = crf.get_unary_term(prob)
        crf_out = crf.crf_post_process(image, unary)

        unary = crf.get_unary_term(cnn_out, unary_from='label', n_classes=2, gt_prob=gt_prob)
        crf_label_out = crf.crf_post_process(image, unary)

        cnn_result_i = my_utils.count_many(cnn_out, label)
        crf_result_i = my_utils.count_many(crf_out, label)
        crf_label_result_i = my_utils.count_many(crf_label_out, label)

        update_dict(cnn_result, cnn_result_i)
        update_dict(crf_result, crf_result_i)
        update_dict(crf_label_result, crf_label_result_i)
    cnn_result = my_utils.metric_many_from_counter(cnn_result)
    crf_result = my_utils.metric_many_from_counter(crf_result)
    crf_label_result = my_utils.metric_many_from_counter(crf_label_result)
    return {
        'cnn': cnn_result,
        'crf': crf_result,
        'crf_label': crf_label_result
    }


def aggregate_result(results):
    final_result = {
        'TP': 0,
        'TN': 0,
        'FP': 0,
        'FN': 0
    }

    def update_dict(target, to_update):
        for key in target:
            target[key] += to_update[key]

    for result in results:
        update_dict(final_result, result)

    final_result.update(my_utils.metric_many_from_counter(final_result))
    logger.info('Final result:\n%s' % json.dumps(final_result, indent=2))


if __name__ == '__main__':
    rs = kfold_evaluation()
    for k, result in rs:
        logger.info(k)
        logger.info('\n%s\n' % json.dumps(result, indent=2))
