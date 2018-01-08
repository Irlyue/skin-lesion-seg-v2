import os
import utils
import numpy as np

from scipy.misc import imread, imresize


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
