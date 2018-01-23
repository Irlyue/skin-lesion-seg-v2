import numpy as np

from pydensecrf.densecrf import DenseCRF2D
from pydensecrf.utils import unary_from_softmax, unary_from_labels


def crf_post_process(image, unary, n_steps=10, sxy=40, srgb=6, compat=1.0):
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
    d.addPairwiseBilateral(sxy=sxy, srgb=srgb, rgbim=image, compat=compat)

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


def crf_from_bbox_circle(image, bbox, n_steps=5):
    """

    :param image:
    :param bbox:
    :param n_steps:
    :return:
    """
    h, w, _ = image.shape
    probs = prob_from_bbox((h, w), bbox)

    unary = get_unary_term(probs)
    result = crf_post_process(image, unary, n_steps=n_steps)
    return result


def prob_from_bbox(shape, bbox):
    def dist(pa, pb):
        xa, ya = pa
        xb, yb = pb
        return np.sqrt((xa - xb)**2 + (ya - yb)**2)

    height, width = shape
    top, left, bh, bw = bbox
    center_h, center_w = top + bh // 2, left + bw // 2
    center = center_h, center_w

    # radius_in = dist(center, (top, left))
    radius_in = min(bh, bw) // 2
    radius_out = min(center_h, height - center_h, center_w, width - center_w)

    p1 = np.zeros(shape)
    for i in range(height):
        for j in range(width):
            dij = dist(center, (i, j))
            if dij < radius_in:
                p1[i, j] = 1.0
    p0 = 1 - p1
    probs = np.stack([p0, p1], axis=2)
    return probs

