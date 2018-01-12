import sys
import inputs
import my_utils

from create_csv import to_csv

logger = my_utils.get_default_logger()


def create_kth_fold(k):
    logger.info('Generating %d-th data for training...' % k)
    config = my_utils.load_config()
    ic = my_utils.load_config('image_config.json')

    dermis = inputs.load_raw_data('dermis', config)
    dermquest = inputs.load_raw_data('dermquest', config)

    train_data = inputs.get_kth_fold(dermquest, k, ic['n_folds'], seed=ic['split_seed'])
    if ic['use_dermis']:
        train_data = train_data + dermis
    train_df = to_csv(train_data)
    train_df.to_csv(ic['train_csv_file'], index=None)
    logger.info('Successfully convert %d examples to %s' % (len(train_df), ic['train_csv_file']))

    test_data = inputs.get_kth_fold(dermquest, k, ic['n_folds'], seed=ic['split_seed'], type_='test')
    test_df = to_csv(test_data)
    test_df.to_csv(ic['test_csv_file'], index=None)
    logger.info('Successfully convert %d examples to %s' % (len(test_df), ic['test_csv_file']))


if __name__ == '__main__':
    # print(sys.argv)
    create_kth_fold(int(sys.argv[1]))
