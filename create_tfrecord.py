import os
import io
import sys
import my_utils
import importlib
import pandas as pd
import tensorflow as tf

from PIL import Image
from collections import namedtuple


sys.path.append('/home/wenfeng/models/models/research')
dataset_util = importlib.import_module('object_detection.utils.dataset_util')
logger = my_utils.get_default_logger()


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'lesion':
        return 1
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(group.filename, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def create_one_record_file(csv_file, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    examples = pd.read_csv(csv_file)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, None)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), output_path)
    logger.info('Successfully created the TFRecords at {}'.format(output_path))


def main(_):
    logger.info('Creating Tf-record files...')
    image_config = my_utils.load_config('./image_config.json')
    create_one_record_file(image_config['train_csv_file'], image_config['train_record_file'])
    create_one_record_file(image_config['test_csv_file'], image_config['test_record_file'])
    logger.info('Tf-record file created!')

if __name__ == '__main__':
    tf.app.run()
