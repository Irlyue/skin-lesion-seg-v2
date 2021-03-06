import os
import my_utils
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

    def __add__(self, other):
        images = self.images + other.images
        labels = self.labels + other.labels
        bboxs = self.bboxs + other.bboxs
        listing = self.listing + other.listing
        return SkinData(images, labels, bboxs, listing)

    def aug_train_batch(self, config):
        batch_size = config['batch_size']
        batch_images, batch_labels, batch_bboxes = [], [], []
        for image, label, bbox in self.aug_train_example(config):
            batch_images.append(image)
            batch_labels.append(label)
            batch_bboxes.append(bbox)
            if len(batch_images) == batch_size:
                yield np.array(batch_images), np.array(batch_labels), np.array(batch_bboxes)
                batch_images, batch_labels, batch_bboxes = [], [], []

    def aug_train_example(self, config):
        n_epochs = config['n_epochs_for_train']
        input_size = config['input_size']
        random = config['random_batch']

        n_examples = len(self.listing)
        max_steps = len(self.listing) * n_epochs
        for i in range(max_steps):
            if random:
                idx = np.random.randint(n_examples)
            else:
                idx = i % n_examples
            new_image, new_label = my_utils.aug_image(*self[idx])
            new_image = imresize(new_image, size=input_size)
            new_label = imresize(new_label, size=input_size, interp='nearest')
            new_bbox = my_utils.calc_bbox(new_label)
            yield new_image, new_label, new_bbox

    def train_batch(self, n_epochs=1, random=True):
        n_examples = len(self.listing)
        max_steps = len(self.listing) * n_epochs
        for i in range(max_steps):
            if random:
                idx = np.random.randint(n_examples)
            else:
                idx = i
            yield self.images[idx], self.labels[idx], self.bboxs[idx]

    def new_train_batch(self, config):
        batch_size = config['batch_size']
        n_epochs = config['n_epochs_for_train']
        batch_images, batch_labels, batch_bboxes = [], [], []
        for image, label, bbox in self.train_batch(n_epochs=n_epochs):
            batch_images.append(image)
            batch_labels.append(label)
            batch_bboxes.append(bbox)
            if len(batch_images) == batch_size:
                yield np.array(batch_images), np.array(batch_labels), np.array(batch_bboxes)
                batch_images, batch_labels, batch_bboxes = [], [], []



def load_training_data(db, config):
    data_dir = config['data_dir']
    ignore_list = load_ignore_list() if config['use_ignore_list'] else []
    images, labels, listing = load_one_database(data_dir, db, ignore_list=ignore_list)
    images = [imresize(image, size=config['input_size']) for image in images]
    labels = [imresize(label, size=config['input_size'], interp='nearest') for label in labels]
    bboxs = calc_bboxs(labels)
    return SkinData(images, labels, bboxs, listing)


def load_raw_data(db, config):
    data_dir = config['data_dir']
    ignore_list = load_ignore_list() if config['use_ignore_list'] else []
    images, labels, listing = load_one_database(data_dir, db, ignore_list=ignore_list)
    bboxs = calc_bboxs(labels)
    return SkinData(images, labels, bboxs, listing)


def load_ignore_list(path=None):
    path = path if path else 'ignore_list.txt'
    with open(path, 'r') as f:
        ignore_list = [item for item in f.read().split('\n') if len(item) > 0]
    return ignore_list


def calc_bboxs(labels):
    return [my_utils.calc_bbox(label) for label in labels]


def get_image_list(data_dir, db):
    path_one = os.path.join(data_dir, 'Skin Image Data Set-1/skin_data/melanoma/')
    path_two = os.path.join(data_dir, 'Skin Image Data Set-2/skin_data/notmelanoma/')
    melanoma = [os.path.join(path_one, db, '_'.join(item.split('_')[:-1]))
                for item in os.listdir(os.path.join(path_one, db)) if item.endswith('.jpg')]
    not_melanoma = [os.path.join(path_two, db, '_'.join(item.split('_')[:-1]))
                    for item in os.listdir(os.path.join(path_two, db)) if item.endswith('.jpg')]
    return melanoma, not_melanoma


def load_one_database(data_dir, db, ignore_list=None):
    """
    Load images from one database.
    :param data_dir: str, directory where the images are located
    :param db: str, database, either `dermis` or `dermquest`.
    :param ignore_list: list, images that should be ignored
    :return:
    """
    cache_path = os.path.join(tempfile.gettempdir(), db + '.pkl')
    if os.path.exists(cache_path):
        images, labels, all_files = my_utils.load_obj(cache_path)
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
        my_utils.dump_obj(cache_path, (images, labels, all_files))
    if ignore_list:
        images, labels, all_files = ignore_some_images(images, labels, all_files, ignore_list)
    return images, labels, all_files


def ignore_some_images(images, labels, all_files, ignore_list):
    filtered = [(image, label, file) for image, label, file in zip(images, labels, all_files)
                if file not in ignore_list]
    images = [item[0] for item in filtered]
    labels = [item[1] for item in filtered]
    all_files = [item[2] for item in filtered]
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


def load_one_example(base, smallest_to=None, highest_to=None, size=None):
    image = imread(base + '_orig.jpg')
    label = imread(base + '_contour.png')
    label[label == 255] = 1
    if smallest_to:
        image = my_utils.imresize_smallest_to(image, smallest_to)
        label = my_utils.imresize_smallest_to(label, smallest_to, method='nearest')
    elif highest_to:
        image = my_utils.imresize_highest_to(image, highest_to)
        label = my_utils.imresize_highest_to(label, highest_to, method='nearest')
    elif size:
        image = imresize(image, size=size)
        label = imresize(label, size=size, interp='nearest')
    bbox = my_utils.calc_bbox(label)
    return image, label, bbox
