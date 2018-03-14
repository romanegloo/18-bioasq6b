#!/usr/bin/env python3
"""BioAsq6B script utilities"""
import time
import random
import logging

logger = logging.getLogger()

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
    """assuming that y_pred is in the order by the confidence score"""
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
        # return ap_num / min(len(y_true), len(y_pred))
        return ap_num / len(y_pred)


def measure_performance(y_true, y_pred, cutoff=None):
    if len(y_pred) > cutoff:
        y_pred = y_pred[:cutoff]
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


class AdaptiveRandomSearch(object):
    def __init__(self, bounds, obj_fn, init_vec=None, \
                 max_iter=100, max_no_impr=3):
        self.bounds = bounds
        self.iter_mult = 5  # 1 in *iter_mul*, use the largest step size
        self.step_sizes = [0.05, 2.0, 3.0]  # initial/small/large factors
        self.step_size = (bounds[0][1] - bounds[0][0]) * self.step_sizes[0]
        self.current = dict()
        if init_vec:
            self.current['vector'] = init_vec
        else:
            self.current['vector'] = self.random_vector()
        self.objective_function = obj_fn
        self.max_iter = max_iter
        self.max_no_impr = max_no_impr  # early stop when no improvement

    @staticmethod
    def rand_in_bounds(min, max):
        return min + ((max - min) * random.random())

    def random_vector(self):
        return [self.rand_in_bounds(self.bounds[i][0], self.bounds[i][1])
                for i in range(len(self.bounds))]

    def large_step_size(self, iter):
        if iter > 0 and iter % self.iter_mult == 0:
            return self.step_size * self.step_sizes[2]
        return self.step_size * self.step_sizes[1]

    def take_step(self, step_size):
        position = []
        v = self.current['vector']
        for i in range(len(v)):
            min_ = max(self.bounds[i][0], v[i] - step_size)
            max_ = min(self.bounds[i][1], v[i] + step_size)
            position.append(self.rand_in_bounds(min_, max_))
        return position

    def take_steps(self, step_size, big_stepsize):
        step = dict()
        big_step = dict()
        step['vector'] = self.take_step(step_size)
        step['cost'] = self.objective_function(step['vector'])
        big_step['vector'] = self.take_step(big_stepsize)
        big_step['cost'] = self.objective_function(big_step['vector'])
        return step, big_step

    def search(self):
        count = 0
        self.current['cost'] = self.objective_function(self.current['vector'])
        for i in range(self.max_iter):
            big_stepsize = self.large_step_size(i)
            # Get the regular and large step size
            step, big_step = self.take_steps(self.step_size, big_stepsize)
            # Compare the costs
            if step['cost'] <= self.current['cost'] or \
                    big_step['cost'] <= self.current['cost']:
                count = 0
                if big_step['cost'] <= self.current['cost']:
                    self.step_size, self.current = big_stepsize, big_step
                else:
                    self.current = step
                logger.info("> updating the weight vector: {}"
                            ''.format(self.current['vector']))
            else:
                count += 1
                if count >= self.max_no_impr:
                    count = 0
                    self.step_size /= self.step_sizes[1]
            logger.info("> iteration {}, best {}, step_size {}"
                        ''.format(i + 1, self.current['cost'], self.step_size))
        return self.current
