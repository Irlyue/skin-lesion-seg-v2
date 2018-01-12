import my_utils
from eval_kfold import OUT_PATH
from kfold_evaluation import aggregate_result

logger = my_utils.get_default_logger()


def aggregate_all_result():
    results = []
    config = my_utils.load_config()
    for i in range(config['n_folds']):
        result_i = my_utils.load_obj(OUT_PATH % i)
        logger.info('***********************Result %d******************' % i)
        logger.info(result_i)
        results.append(result_i)
    final_result = aggregate_result(results)
    logger.info('**********************Final Result********************')
    logger.info(final_result)


if __name__ == '__main__':
    aggregate_all_result()
