#!/usr/bin/env python3
"""BioAsq6B script utilities"""
import time

# ------------------------------------------------------------------------------
# Evaluation measure functions
# ------------------------------------------------------------------------------

def precision(y_true, y_pred):
    """p = tp / (tp + fp)"""
    tp = len([x for x in y_pred if x in y_true])
    return tp / len(y_pred) if len(y_pred) != 0 else 0.


def recall(y_true, y_pred):
    """r = tp / (tp + fn)"""
    tp = len([x for x in y_pred if x in y_true])
    fn = len(y_true) - tp
    return tp / (tp + fn) if (tp + fn) != 0 else 0.


def f_measure(prec, recall, beta=1):
    b2 = beta ** 2
    if prec == 0 and recall == 0:
        return 0.
    return (1 + b2) * prec * recall / (b2 * prec + recall)


def avg_precision(y_true, y_pred):
    """assuming that y_pred is in order by the confidence score"""
    ap_num = 0.
    cnt_matches = 0
    for idx, item in enumerate(y_pred):
        if item in y_true:
            cnt_matches += 1
            ap_num += cnt_matches / (idx + 1)

    # avoid division by zero
    if cnt_matches == 0:
        return 1 if len(y_true) == 0 else 0
    else:
        return ap_num / min(len(y_true), len(y_pred))

def measure_performance(y_true, y_pred):
    precision_ = precision(y_true, y_pred)
    recall_ = recall(y_true, y_pred)
    F1 = f_measure(precision_, recall_)
    avg_prec = avg_precision(y_true, y_pred)
    return precision_, recall_, F1, avg_prec


# ------------------------------------------------------------------------------
# Utility classes
# ------------------------------------------------------------------------------


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total
