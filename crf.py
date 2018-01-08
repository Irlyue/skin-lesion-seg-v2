import numpy as np

from pydensecrf.densecrf import DenseCRF2D
from pydensecrf.utils import unary_from_softmax, unary_from_labels


def crf_post_process(image, unary, n_steps=5):
    """
    Perform CRF post process giving the unary.

    :param image: np.array, with shape(height, width, 3)
    :param unary: np.array, with shape(n_classes, height * width)
    :param n_steps: int, number of iteration
    :return:
        result: np.array, with shape(height, width) given the segmentation mask.
    """
    height, width, _ = image.shape
    n_classes = unary.shape[0]
    d = DenseCRF2D(width, height, n_classes)

    # unary potential
    d.setUnaryEnergy(unary)

    # pairwise potential
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=2)

    # inference
    Q = d.inference(n_steps)
    result = np.argmax(Q, axis=0).reshape((height, width))
    return result


def get_unary_term(x, unary_from='prob', n_classes=None, gt_prob=None):
    """
    Get unary potential either from probability or label guess. Basically, it can be used
    as follows:
    1) get_unary_term(prob);                 # from probability
    2) get_unary_term(label_guess,
                      unary_from='label',
                      n_classes=2,
                      gt_prob=0.7)           # from label guess

    :param x: np.array, the unary source with shape(height, width) or (height, width, n_classes)
    :param unary_from: str, either 'prob' or 'label'
    :param n_classes: int, number of classes
    :param gt_prob: float, between (0.0, 1.0), giving the confidence about the label.
    :return:
    """
    if unary_from == 'prob':
        unary = unary_from_softmax(x.transpose((2, 0, 1)))
    elif unary_from == 'label':
        unary = unary_from_labels(x, n_classes,
                                  gt_prob=gt_prob,
                                  zero_unsure=False)
    else:
        raise NotImplemented
    return unary


def crf_from_bbox(image, bbox, gt_prob, n_steps=5):
    """
    Perform CRF post-processing giving the bounding-box. Pixels inside the bounding-box
    are labeled as positive, otherwise as negative. Then this label guess is refined by
    the CRF process.

    :param image: np.array, with shape(height, width, 3)
    :param bbox: tuple or list, with shape(4,) giving the bounding-box position[top, left,
    height, width]
    :param gt_prob: float, between (0.0, 1.0), confidence about the label
    :param n_steps: int, number of iteration
    :return:
        result: np.array, 
    """
    h, w, _ = image.shape
    top, left, height, width = bbox
    label_guess = np.zeros((h, w), dtype=np.int32)
    label_guess[top:top + height, left:left + width] = 1

    unary = get_unary_term(label_guess, unary_from='label', n_classes=2, gt_prob=gt_prob)
    result = crf_post_process(image, unary, n_steps=n_steps)
    return result
