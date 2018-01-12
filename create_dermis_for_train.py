import inputs
import my_utils

from create_csv import to_csv

logger = my_utils.get_default_logger()


def create_dermis_for_train_dermquest_for_test():
    logger.info('Creating dermis for training and dermquest for testing...')
    config = my_utils.load_config()
    image_config = my_utils.load_config('image_config.json')
    dermis = inputs.load_raw_data('dermis', config)
    dermquest = inputs.load_raw_data('dermquest', config)

    train_df = to_csv(dermis)
    train_df.to_csv(image_config['train_csv_file'], index=None)
    logger.info('Successfully convert %d examples to %s' % (len(train_df), image_config['train_csv_file']))

    test_df = to_csv(dermquest)
    test_df.to_csv(image_config['test_csv_file'], index=None)
    logger.info('Successfully convert %d examples to %s' % (len(test_df), image_config['test_csv_file']))


if __name__ == '__main__':
    create_dermis_for_train_dermquest_for_test()
