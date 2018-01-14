import json
import inputs
import my_utils
import evaluation
import tensorflow as tf

from kfold_seg_evaluation import cnn_eval_one, crf_eval_one, crf_label_eval_one


logger = my_utils.get_default_logger()


def eval_seg_model(eval_one_func):
    logger.info('K-fold evaluation process...')
    config = my_utils.load_config()
    dermquest = inputs.load_raw_data('dermquest', config)

    mm = evaluation.SegRestoredModel(tf.train.latest_checkpoint(config['train_dir']))
    result = test_one_model(mm, dermquest.listing, config, eval_one_func)
    logger.info('Done evaluation')
    return result


def test_one_model(model, listing, config, eval_one_func):
    result = {
        'TP': 0,
        'TN': 0,
        'FP': 0,
        'FN': 0
    }

    logger.info('Evaluating %d examples in total...' % len(listing))

    def update_dict(target, to_update):
        for key in to_update:
            target[key] += to_update[key]

    for base in listing:
        image, label, bbox = inputs.load_one_example(base, size=config['input_size'])
        result_i = eval_one_func(model, image, label)
        update_dict(result, result_i)
    result = my_utils.metric_many_from_counter(result)
    return result


def eval_many_methods(eval_funcs):
    eval_results = {}
    for key, eval_func in eval_funcs.items():
        logger.info('-----------------------> %s' % key)
        results = eval_seg_model(eval_func)
        eval_results[key] = results
    return eval_results


def display_results(results):
    for key, result in results.items():
        logger.info('----------------------> %s' % key)
        logger.info('\n----> Result <----\n%s' % json.dumps(result, indent=2))


if __name__ == '__main__':
    eval_funcs = {
        'cnn': cnn_eval_one,
        'crf': crf_eval_one,
        'crf_label': lambda mm, image, label: crf_label_eval_one(mm, image, label, gt_prob=0.9)
    }
    display_results(eval_many_methods(eval_funcs))
