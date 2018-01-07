import numpy as np

from pydensecrf.densecrf import DenseCRF2D
from pydensecrf.utils import unary_from_softmax, unary_from_labels


def crf_post_process(image, x, unary_from='prob', n_steps=5, gt_prob=0.6):
    """
    Use CRF as a post processing technique. Basically, it can be used as follows:
        1. crf_post_process(image, probability)
        2. crf_post_process(image, label_guess, unary_from='label')

    :param image: np.array, the raw image with shape like(height, width, n_classes)
    :param x: np.array, same shape as `image`, giving the unary source, either probability or label.
    :param n_steps: int, number of iterations for CRF inference.
    :param unary_from: str, either 'prob' or 'label', specifying the unary type.
    :param gt_prob: float, between(0, 1), only useful when unary_from equals to 'label'.
    :return:
        result: np.array(dtype=np.int32), result after the CRF post-processing.
    """
    assert image.shape[:2] == x.shape[:2], '<<ERROR>>Image shape%s not equal to x\'s shape%s' % (image.shape, x.shape)
    height, width, n_classes = x.shape
    d = DenseCRF2D(width, height, n_classes)

    # unary potential
    unary = get_unary_term(x, unary_from, gt_prob=gt_prob, n_labels=n_classes)
    d.setUnaryEnergy(unary)

    # pairwise potential
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=2)

    # inference
    Q = d.inference(n_steps)
    result = np.argmax(Q, axis=0).reshape((height, width))
    return result


def get_unary_term(x, unary_from='prob', n_labels=None, gt_prob=None):
    if unary_from == 'prob':
        unary = unary_from_softmax(x.transpose((2, 0, 1)))
    elif unary_from == 'label':
        unary = unary_from_labels(x, n_labels=n_labels, gt_prob=gt_prob)
    else:
        raise NotImplemented
    return unary
