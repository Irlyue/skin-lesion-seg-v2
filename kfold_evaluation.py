import os
import crf
import json
import utils
import inputs
import evaluation
import tensorflow as tf


logger = utils.get_default_logger()


def kfold_evaluation():
    final_result = []
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
        final_result.append(result)
        logger.info('Result for %d-th: \n%s' % (i, json.dumps(result, indent=2)))
        logger.info('************************************\n\n')
    logger.info('Done evaluation')
    return final_result


def evaluate_one_fold(data, config):
    with tf.Graph().as_default():
        model = evaluation.EvalBboxModel(config)
        result = evaluation.evaluate_one_model(model, data, config)
    return result


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

    final_result.update(utils.metric_many_from_counter(final_result))
    logger.info('Final result:\n%s' % json.dumps(final_result, indent=2))


if __name__ == '__main__':
    rs = kfold_evaluation()
    aggregate_result(rs)
