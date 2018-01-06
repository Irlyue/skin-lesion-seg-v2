import os
import utils

from scipy.misc import imread


class SkinData:
    def __init__(self, images, labels, bboxs, listing=None):
        self.images = images
        self.labels = labels
        self.bboxs = bboxs
        self.listing = listing

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.bboxs[idx]


def load_training_data(db, config):
    data_dir = config['data_dir']
    images, labels, listing = load_one_database(data_dir, db)
    bboxs = calc_bboxs(labels)
    return SkinData(images, labels, bboxs, listing)


def calc_bboxs(labels):
    return [utils.calc_bbox(label) for label in labels]


def get_image_list(data_dir, db):
    path_one = os.path.join(data_dir, 'Skin Image Data Set-1/skin_data/melanoma/')
    path_two = os.path.join(data_dir, 'Skin Image Data Set-2/skin_data/notmelanoma/')
    melanoma = [os.path.join(path_one, db, item.split('_')[0])
                for item in os.listdir(os.path.join(path_one, db)) if not item.endswith('db')]
    not_melanoma = [os.path.join(path_two, db, item.split('_')[0])
                    for item in os.listdir(os.path.join(path_two, db)) if not item.endswith('db')]
    return melanoma, not_melanoma


def load_one_database(data_dir, db):
    images = []
    labels = []
    melanoma, not_melanoma = get_image_list(data_dir, db)
    all_files = melanoma + not_melanoma
    for item in all_files:
        image_path = item + '_orig.jpg'
        label_path = item + '_contour.png'
        image = imread(image_path)
        label = imread(label_path)
        label[label == 255] = 1
        images.append(image)
        labels.append(label)
    return images, labels, all_files
