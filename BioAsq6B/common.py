#!/usr/bin/env python3
"""BioAsq6B script utilities"""
import time
import random
import logging
import prettytable
import re
from termcolor import colored
import math
import numpy as np
import subprocess
from multiprocessing import Pool
from functools import partial

from BioAsq6B import PATHS

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


def avg_precision(y_true, y_pred, cutoff=10):
    """ Average Precision at cutoff"""
    if len(y_pred) > cutoff:
        y_pred = y_pred[:cutoff]

    ap_numerator = 0.
    cnt_matches = 0
    for idx, item in enumerate(y_pred):
        if item in y_true and item not in y_pred[:idx]:
            cnt_matches += 1
            ap_numerator += cnt_matches / (idx + 1)

    if not y_true:
        return 0.

    return ap_numerator / min(len(y_true), cutoff)


def measure_performance(y_true, y_pred, cutoff=10):
    precision_ = precision(y_true, y_pred)
    recall_ = recall(y_true, y_pred)
    F1 = f_measure(precision_, recall_)
    avg_prec = avg_precision(y_true, y_pred[:cutoff])
    return precision_, recall_, F1, avg_prec


# ------------------------------------------------------------------------------
# Utility classes
# ------------------------------------------------------------------------------


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def __repr__(self):
        return str(self.avg)

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
    def __init__(self, bounds, obj_fn, init_vec=None, weight_keys=None,
                 data=None, max_iter=100, max_no_impr=5):
        self.data = data
        self.bounds = bounds
        self.iter_mult = 5  # once in *iter_mul*, use the largest step size
        self.step_sizes = [1.00, 1.5, 3.0]  # initial/small/large factors
        self.step_size = (bounds[0][1] - bounds[0][0]) * self.step_sizes[0]
        self.current = dict()
        if init_vec is not None:
            self.current['vector'] = init_vec
        else:
            self.current['vector'] = self.random_vector()
        if weight_keys is not None:
            self.objective_function = partial(obj_fn, weight_keys=weight_keys)
        else:
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
        # Normalize
        return [w/sum(position) for w in position]

    def take_steps(self, step_size, big_stepsize):
        step = dict()
        big_step = dict()
        step['vector'] = self.take_step(step_size)
        step['cost'] = self.objective_function(step['vector'], self.data)
        big_step['vector'] = self.take_step(big_stepsize)
        big_step['cost'] = \
            self.objective_function(big_step['vector'], self.data)
        return step, big_step

    def search(self):
        count = 0
        self.current['cost'] = \
            self.objective_function(self.current['vector'], self.data)
        for i in range(self.max_iter):
            big_stepsize = self.large_step_size(i)
            # Get the regular and large step size
            step, big_step = self.take_steps(self.step_size, big_stepsize)
            # Compare the costs
            if step['cost'] < self.current['cost'] or \
                    big_step['cost'] < self.current['cost']:
                count = 0
                if big_step['cost'] < self.current['cost']:
                    self.step_size, self.current = big_stepsize, big_step
                else:
                    self.current = step
                logger.info("> UPDATING WEIGHT VECTOR {}"
                            ''.format(self.current['vector']))
            else:
                count += 1
                if count >= self.max_no_impr:
                    count = 0
                    self.step_size /= self.step_sizes[1]
            logger.info("> iteration {}, best {:.4f}, step_size {:.4e}"
                        ''.format(i + 1, self.current['cost'], self.step_size))
            if self.step_size < 1e-4:
                logger.info('> early stop (step_size {})'
                            ''.format(self.step_size))
                break
        return self.current


class RankedDocs(object):
    """A container for the ranked list of documents with scores and auxiliary
    data entities"""
    def __init__(self, query):
        self.query = query
        self.rankings = []  # Ranked docs by retrieval score
        self.rankings_fusion = []  # Final ranks after score fusion
        self.scores = dict()  # Different kinds of scores including retrieval
        self.docs_data = dict()
        self.expected_docs = []
        self.unseen_words = set()
        self.text_snippets = []
        # Read expected documents (GT) if exist
        if 'documents' in self.query:
            for line in self.query['documents']:
                m = re.match(".*pubmed/(\d+)$", line)
                if m:
                    self.expected_docs.append(m.group(1))

    @staticmethod
    def galago_read_doc(docid):
        try:
            p = subprocess.run(
                ['galago', 'doc', '--index={}'.format(PATHS['galago_idx']),
                 '--id=PMID-{}'.format(docid)],
                stdout=subprocess.PIPE
            )
            doc = p.stdout.decode('utf-8')
            # find the fields in the raw text
            fields = {'text': 'TEXT', 'title': 'TITLE', 'journal': 'MEDLINE_TA'}
            doc_ = dict()
            doc_['docid'] = docid
            for k, v in fields.items():
                tag_len = len('<{}>'.format(v))
                start = doc.find('<{}>'.format(v)) + tag_len
                end = doc.find('</{}>'.format(v))
                if start >= end or start <= tag_len or end <= 0:
                    contents = ''
                else:
                    contents = doc[start:end]
                doc_[k] = contents
        except:
            logger.error(docid)
            raise
        return doc_

    def read_doc_text(self):
        """Retrieve document in the rankings list from the galago index"""
        def cb_galago_read_doc(res):
            self.docs_data[res['docid']] = res
        p = Pool(30)
        for d in self.rankings:
            p.apply_async(self.galago_read_doc, args=(d, ),
                          callback=cb_galago_read_doc)
        p.close()
        p.join()

    def write_result(self, printout=False, stats=None, topn=10, seq=None):
        """Print out the results of a query with the performance measures"""
        # Set tabular format for eval results
        table = prettytable.PrettyTable(['Question', 'GT', 'Returned',
                                         'Scores'])
        table.align['Question'] = 'l'
        table.align['Returned'] = 'r'
        table.max_width['Question'] = 40
        col0 = '[{}]\n{}'.format(self.query['id'], self.query['body'])
        if 'ideal_answer' in self.query:
            col0 += '\n=> {}'.format(self.query['ideal_answer'])
        col1 = []
        for docid in self.expected_docs:
            docid_ = colored(docid, 'yellow') \
                if docid in self.rankings_fusion else docid
            col1.append(docid_)
        col1 = '\n'.join(col1)
        col2 = []  # Returned documents
        col3 = []  # Scores
        for docid in self.rankings_fusion[:topn]:
            docid_ = \
                colored(docid, 'blue') if docid in self.expected_docs else docid
            col2.append('{:>8}'.format(docid_))
            idx = self.rankings.index(docid)
            scores_ = "{:.2f}".format(self.scores['retrieval'][idx])
            if 'qasim' in self.scores:
                scores_ += "/{:.2f}".format(np.max(self.scores['qasim'][idx]))
            if 'journal' in self.scores:
                scores_ += "/{:.2f}".format(np.max(self.scores['journal'][idx]))
            if 'semmeddb' in self.scores:
                scores_ += '/({}/{:.2f})' \
                           ''.format(self.scores['semmeddb'][idx][0],
                                     self.scores['semmeddb'][idx][1])
            scores_ = "{:.4f} ({})".format(self.scores['fusion'][idx], scores_)
            col3.append(scores_)
        col2 = '\n'.join(col2)
        col3 = '\n'.join(col3)
        table.add_row([col0, col1, col2, col3])
        if len(self.expected_docs) > 0 and stats is not None:
            prec, recall, f1, ap = \
                measure_performance(self.expected_docs, self.rankings_fusion,
                                    cutoff=topn)
            # Update statistics
            stats['avg_prec'].update(prec)
            stats['avg_recall'].update(recall)
            stats['avg_f1'].update(f1)
            stats['map'].update(ap)
            stats['logp'].update(math.log(prec + 1e-6))
            report = ('precision: {:.4f}, recall: {:.4f}, F1: {:.4f}, '
                      'avg_prec_10: {:.4f}').format(prec, recall, f1, ap)
        else:
            report = ''
        if printout:
            logger.info("Details:\n" + table.get_string() + "\n")
            if seq is not None:
                line = '[seq. {}/{}] {}'.format(*seq, report)
            else:
                line = '{}'.format(report)
            logger.info(line)
        return

    def merge_scores(self, weights_str):
        # Merge different set of scores into "fusion" scores
        ndocs = len(self.scores['retrieval'])
        weights = {}
        for w in weights_str.split(','):
            k, v = w.split(':')
            if k == 'retrieval':
                weights['retrieval'] = float(v)
            elif k == 'qasim':
                weights['qasim'] = float(v)
            elif k == 'journal':
                weights['journal'] = float(v)
            elif k == 'semmeddb1':
                weights['semmeddb1'] = float(v)
            elif k == 'semmeddb2':
                weights['semmeddb2'] = float(v)

        # Initialize rankings_fusion with retrieval scores
        self.rankings_fusion = self.rankings
        # Normalize the retrieval scores
        ret_norm = list(map(lambda x: math.exp(x), self.scores['retrieval']))
        ret_norm = [float(s)/sum(ret_norm) for s in ret_norm]
        self.scores['fusion'] = ret_norm
        if len(weights.keys()) == 0:
            return
        fusion = weights['retrieval'] * ndocs * np.array(ret_norm)
        if 'qasim' in weights and 'qasim' in self.scores:
            for i, score in enumerate(self.scores['qasim']):
                fusion[i] += sum(sorted(score)[-3:]) / 3 * weights['qasim']
        if 'journal' in weights and 'journal' in self.scores:
            fusion += np.array(self.scores['journal']) * weights['journal']
        if 'semmeddb1' in weights and 'semmeddb2' in weights \
            and 'semmeddb' in self.scores:
            for i, scores in enumerate(self.scores['semmeddb']):
                fusion[i] += scores[0] * weights['semmeddb1'] + \
                             scores[1] * weights['semmeddb2']
        self.scores['fusion'] = fusion.tolist()
        # Order by fusion scores
        self.rankings_fusion = \
            [docid for _, docid
             in sorted(zip(self.scores['fusion'], self.rankings),
                       key=lambda pair: pair[0], reverse=True)]


# Commonly used functions
def keys_exists(element, *keys):
    '''
    Check if *keys (nested) exists in `element` (dict).
    '''
    if type(element) is not dict:
        raise AttributeError('keys_exists() expects dict as first argument.')
    if len(keys) == 0:
        raise AttributeError('keys_exists() expects at least two arguments, one given.')

    _element = element
    for key in keys:
        try:
            _element = _element[key]
        except KeyError:
            return False
    return True


def softmax(lst):
    exp_lst = np.exp(lst - np.max(lst))
    norm = exp_lst / exp_lst.sum(axis=0)
    if type(lst) == list:
        return norm.tolist()
    else:
        return norm
