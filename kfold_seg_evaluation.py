import os
import crf
import json
import inputs
import my_utils
import evaluation
import numpy as np
import tensorflow as tf


logger = my_utils.get_default_logger()


def cnn_eval_one(mm, image, label):
    out = mm.inference(image, ['mask'])[0]
    out = np.squeeze(out)
    result = my_utils.count_many(predictions=out,
                                labels=label)
    return result


def crf_eval_one(mm, image, label):
    out, prob = mm.inference(image, ['mask', 'prob'])
    out, prob = np.squeeze(out), np.squeeze(prob)
    unary = crf.get_unary_term(prob)
    crf_out = crf.crf_post_process(image, unary)
    result = my_utils.count_many(predictions=crf_out,
                                 labels=label)
    return result


def crf_label_eval_one(mm, image, label, gt_prob):
    out, prob = mm.inference(image, ['mask', 'prob'])
    out, prob = np.squeeze(out), np.squeeze(prob)

    unary = crf.get_unary_term(out, unary_from='label', gt_prob=gt_prob, n_classes=2)
    crf_label_out = crf.crf_post_process(image, unary)
    result = my_utils.count_many(predictions=crf_label_out,
                                 labels=label)
    return result


def cnn_hole_filling_eval_one(mm, image, label):
    out = mm.inference(image, ['mask'])[0]
    out = np.squeeze(out)
    hole_fill_out, _ = my_utils.hole_filling(out)
    result = my_utils.count_many(predictions=hole_fill_out,
                                 labels=label)
    return result


def crf_hole_filling_eval_one(mm, image, label):
    out, prob = mm.inference(image, ['mask', 'prob'])
    out, prob = np.squeeze(out), np.squeeze(prob)

    unary = crf.get_unary_term(prob)
    crf_label_out = crf.crf_post_process(image, unary)
    hole_fill_out, _ = my_utils.hole_filling(crf_label_out)
    result = my_utils.count_many(predictions=hole_fill_out,
                                 labels=label)
    return result


def kfold_evaluation(eval_one_func):
    results = []
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
        result = test_one_model(mm, test_data.listing, kfold_config, eval_one_func)
        results.append(result)
        logger.info('************************************\n\n')
    logger.info('Done evaluation')
    return results


def test_one_model(model, listing, config, eval_one_func):
    result = {
        'TP': 0,
        'TN': 0,
        'FP': 0,
        'FN': 0
    }

    def update_dict(target, to_update):
        for key in to_update:
            target[key] += to_update[key]

    for base in listing:
        image, label, bbox = inputs.load_one_example(base, size=config['input_size'])
        result_i = eval_one_func(model, image, label)
        update_dict(result, result_i)
    result = my_utils.metric_many_from_counter(result)
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

    final_result.update(my_utils.metric_many_from_counter(final_result))
    logger.info('Final result:\n%s' % json.dumps(final_result, indent=2))


def mean_aggregate_result(results):
    final_result = {}
    n_folds = len(results)
    for key in results[0]:
        final_result[key] = sum(result[key] for result in results) * 1.0 / n_folds
    return final_result


def eval_many_methods(eval_funcs):
    eval_results = {}
    for key, eval_func in eval_funcs.items():
        logger.info('-----------------------> %s' % key)
        results = kfold_evaluation(eval_func)
        eval_results[key] = results
    return eval_results


def display_results(results):
    def display_one(result):
        final_result = mean_aggregate_result(result)
        logger.info('\n'.join('%s' % json.dumps(item, indent=2) for item in result))
        logger.info('\n---->Final Result<----\n%s' % json.dumps(final_result, indent=2))
    for key, result in results.items():
        logger.info('----------------------> %s' % key)
        display_one(result)


if __name__ == '__main__':
    eval_funcs = {
        'cnn': cnn_eval_one,
        'crf': crf_eval_one
        # 'cnn_hole_filling': cnn_hole_filling_eval_one
    }
    display_results(eval_many_methods(eval_funcs))
