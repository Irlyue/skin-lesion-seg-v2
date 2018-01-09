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
    with tf.Graph().as_default():
        model = evaluation.EvalModel(config)
        result = evaluation.evaluate_one_model(model, data, config)
    return result


if __name__ == '__main__':
    kfold_evaluation()
