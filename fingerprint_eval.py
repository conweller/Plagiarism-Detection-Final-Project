import fingerprint
import itertools

from dataset import *

def precision(pred, gt):
    tp = len(set.intersection(pred, gt))
    fp = len(pred-gt)
    if (tp + fp) == 0:
        return float("nan")
    else:
        return tp / (tp + fp)


def recall(pred, gt):
    tp = len(set.intersection(pred, gt))
    fn = len(gt - pred)
    if (tp + fn) == 0:
        return float("nan")
    else:
        return tp / (tp + fn)


def f1(pred, gt):
    p = precision(pred, gt)
    r = recall(pred, gt)
    if (p + r) == 0:
        return float("nan")
    else:
        return 2 * p * r / (p + r)


def accuracy(pred, gt, count):
    tp = len(set.intersection(pred, gt))
    fp = len(pred-gt)
    tn = count - tp - fp
    return (tp + tn) / count


def report_performance(assignments, t=5, k=100, w=100):
    recalls = []
    precisions = []
    accuracies = []
    f1s = []
    for assignment in assignments:
        pred = set(itertools.chain.from_iterable(list(s)
                   for s in fingerprint.matches(assignment, t=t, k=k, w=w)))
        gt = set(itertools.chain.from_iterable(list(s)
                   for s in assignment.plagiarism_pairs))
        recalls.append(recall(pred, gt))
        precisions.append(precision(pred, gt))
        f1s.append(f1(pred, gt))
        # count = len(list(itertools.combinations(assignment.ids, r=2)))
        count = len(assignment.ids)
        accuracies.append(accuracy(pred, gt, count))
        print(accuracies[-1])
    recalls = [r for r in recalls if r == r]
    precisions = [r for r in precisions if r == r]
    accuracies = [r for r in accuracies if r == r]
    f1s = [r for r in f1s if r == r]
    print((t, k, w))
    print(f"\trecall: {sum(recalls) / len(recalls)}")
    print(f"\tprecision: {sum(precisions) / len(precisions)}")
    print(f"\tf1: {sum(f1s) / len(f1s)}")
    print(f"\taccuracy: {sum(accuracies) / len(accuracies)}")


def fingerprint_hyper_params(assignments):
    # folds = leave_one_out_folds(assignments)
    params = {"t": [1, 3, 5],
              "k": [10, 50, 100],
              "w": [50, 100, 150]}
    for (t, k, w) in itertools.product(*params.values()):
        report_performance(assignments, t=t, k=k, w=w)


if __name__ == "__main__":
    fingerprint_hyper_params(training)
