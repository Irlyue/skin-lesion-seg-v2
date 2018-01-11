import my_utils
import inputs
import pandas as pd


logger = my_utils.get_default_logger()


def to_csv(data):
    def bbox_to_xy(box):
        top_in, left_in, height_in, width_in = box
        return left_in, top_in, left_in + width_in, top_in + height_in

    values = []
    for i, (image, label, bbox) in enumerate(data):
        path = data.listing[i]
        file_path = path + '_orig.jpg'
        height, width, _ = image.shape
        xmin, ymin, xmax, ymax = bbox_to_xy(bbox)
        value = (file_path, width, height, 'lesion', xmin, ymin, xmax, ymax)
        values.append(value)
    column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    df = pd.DataFrame(values, columns=column_names)
    return df


def main():
    logger.info('Creating csv file...')
    config = my_utils.load_config()
    image_config = my_utils.load_config('./image_config.json')
    k = image_config['k']
    n_folds = image_config['n_folds']
    split_seed = image_config['split_seed']
    use_dermis = image_config['use_dermis']
    train_csv_file = image_config['train_csv_file']
    test_csv_file = image_config['test_csv_file']

    dermquest = inputs.load_raw_data('dermquest', config)
    dermis = inputs.load_raw_data('dermis', config)

    train_data = inputs.get_kth_fold(dermquest, k, n_folds, seed=split_seed)
    if use_dermis:
        train_data = train_data + dermis
    train_df = to_csv(train_data)
    train_df.to_csv(train_csv_file, index=None)
    logger.info('Successfully convert %d examples to %s' % (len(train_data), train_csv_file))

    test_data = inputs.get_kth_fold(dermquest, k, n_folds, seed=split_seed, type_='test')
    test_df = to_csv(test_data)
    test_df.to_csv(test_csv_file, index=None)
    logger.info('Successfully convert %d examples to %s' % (len(test_data), test_csv_file))
    logger.info('CSV file created!')


if __name__ == '__main__':
    main()
