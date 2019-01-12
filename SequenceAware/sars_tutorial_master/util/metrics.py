import numpy as np

def precision(ground_truth, prediction):
    """
    Compute Precision metric
    :param ground_truth: the ground truth set or sequence
    :param prediction: the predicted set or sequence
    :return: the value of the metric
    """
    ground_truth = remove_duplicates(ground_truth)
    prediction = remove_duplicates(prediction)
    precision_score = count_a_in_b_unique(prediction, ground_truth) / float(len(prediction))
    assert 0 <= precision_score <= 1
    return precision_score


def recall(ground_truth, prediction):
    """
    Compute Recall metric
    :param ground_truth: the ground truth set or sequence
    :param prediction: the predicted set or sequence
    :return: the value of the metric
    """
    ground_truth = remove_duplicates(ground_truth)
    prediction = remove_duplicates(prediction)
    recall_score = 0 if len(prediction) == 0 else count_a_in_b_unique(prediction, ground_truth) / float(
        len(ground_truth))
    assert 0 <= recall_score <= 1
    return recall_score


def mrr(ground_truth, prediction):
    """
    Compute Mean Reciprocal Rank metric. Reciprocal Rank is set 0 if no predicted item is in contained the ground truth.
    :param ground_truth: the ground truth set or sequence
    :param prediction: the predicted set or sequence
    :return: the value of the metric
    """
    ground_truth = remove_duplicates(ground_truth)
    prediction = remove_duplicates(prediction)
    rr = 0.
    for rank, p in enumerate(prediction):
        if p in ground_truth:
            rr = 1. / (rank + 1)
            break
    return rr

def mean_average_precision(ground_truth, prediction):
    """
    Compute ...
    :param ground_truth: the ground truth set or sequence
    :param prediction: the predicted set or sequence
    :return: the value of the metric
    """
    ground_truth = remove_duplicates(ground_truth)
    prediction = remove_duplicates(prediction)
    # Cumulative sum: precision at 1, at 2, at 3 ...
    is_relevant = np.in1d(prediction, ground_truth, assume_unique=True)
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    map_score = np.sum(p_at_k) / np.min([ground_truth.shape[0], is_relevant.shape[0]])
    return map_score


def count_a_in_b_unique(a, b):
    """
    :param a: list of lists
    :param b: list of lists
    :return: number of elements of a in b
    """
    count = 0
    for el in a:
        if el in b:
            count += 1
    return count


def remove_duplicates(l):
    return [list(x) for x in set(tuple(x) for x in l)]
