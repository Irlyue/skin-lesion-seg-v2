import os
import utils
import tempfile
import numpy as np

from scipy.misc import imread, imresize
from sklearn.model_selection import KFold


class SkinData:
    def __init__(self, images, labels, bboxs, listing=None):
        self.images = images
        self.labels = labels
        self.bboxs = bboxs
        self.listing = listing

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.bboxs[idx]

    def train_batch(self, n_epochs=1, random=True):
        n_examples = len(self.listing)
        max_steps = len(self.listing) * n_epochs
        for i in range(max_steps):
            if random:
                idx = np.random.randint(n_examples)
            else:
                idx = i
            yield self.images[idx], self.labels[idx], self.bboxs[idx]


def load_training_data(db, config):
    data_dir = config['data_dir']
    images, labels, listing = load_one_database(data_dir, db)
    images = [imresize(image, size=config['input_size']) for image in images]
    labels = [imresize(label, size=config['input_size'], interp='nearest') for label in labels]
    bboxs = calc_bboxs(labels)
    return SkinData(images, labels, bboxs, listing)


def load_raw_data(db, config):
    data_dir = config['data_dir']
    images, labels, listing = load_one_database(data_dir, db)
    bboxs = calc_bboxs(labels)
    return SkinData(images, labels, bboxs, listing)


def calc_bboxs(labels):
    return [utils.calc_bbox(label) for label in labels]


def get_image_list(data_dir, db):
    path_one = os.path.join(data_dir, 'Skin Image Data Set-1/skin_data/melanoma/')
    path_two = os.path.join(data_dir, 'Skin Image Data Set-2/skin_data/notmelanoma/')
    melanoma = [os.path.join(path_one, db, '_'.join(item.split('_')[:-1]))
                for item in os.listdir(os.path.join(path_one, db)) if item.endswith('.jpg')]
    not_melanoma = [os.path.join(path_two, db, '_'.join(item.split('_')[:-1]))
                    for item in os.listdir(os.path.join(path_two, db)) if item.endswith('.jpg')]
    return melanoma, not_melanoma


def load_one_database(data_dir, db):
    cache_path = os.path.join(tempfile.gettempdir(), db + '.pkl')
    if os.path.exists(cache_path):
        images, labels, all_files = utils.load_obj(cache_path)
    else:
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
        utils.dump_obj(cache_path, (images, labels, all_files))
    return images, labels, all_files


def get_kth_fold(data, k, n_folds, seed=None, type_='train'):
    """
    Function for k-fold cross-validation.

    :param data: SkinData, the data source to split
    :param k: int, k-th fold
    :param n_folds: int, number of splits
    :param seed: int, optional, whether to shuffle the input data before split
    :param type_: str, either 'train' or 'test', specifying the split indexes to return
    :return:
        result: SkinData, k-th fold data
    """
    assert type_ == 'train' or type_ == 'test', "Choose from 'train' or 'test' for parameter `type_`"
    shuffle = (seed is not None)
    kf = KFold(n_splits=n_folds, random_state=seed, shuffle=shuffle)
    train_idx, test_idx = list(kf.split(data))[k]
    idx = train_idx if type_ == 'train' else test_idx
    images = [data.images[i] for i in idx]
    labels = [data.labels[i] for i in idx]
    bboxs = [data.bboxs[i] for i in idx]
    listing = [data.listing[i] for i in idx]
    # return idx
    return SkinData(images, labels, bboxs, listing)