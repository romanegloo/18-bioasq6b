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
            if self.step_size < 1e-5:
                logger.info("> early stop (step_size {})"
                             "".format(self.step_size))
                break
        return self.current


class RankedDocs(object):
    """A container for the reanked list of documents with scores and auxilary
    data entities"""
    def __init__(self, query):
        self.query = query
        self.rankings = []  # Ranked docs by retrieval score
        self.rankings_fusion = []  # Final ranks after score fusion
        self.scores = dict()  # Different kinds of scores including retrieval
        self.docs_data = dict()
        self.expected_docs = []
        self.update_cache = []  # If not empty, Cache will update the scores
        # Read expected documents (GT) if exist
        if 'documents' in self.query:
            for line in self.query['documents']:
                m = re.match(".*pubmed/(\d+)$", line)
                if m:
                    self.expected_docs.append(m.group(1))

    def read_doc_text(self, cache=None):
        """Retrieve document in the rankings list from the galago index"""
        for docid in self.rankings:
            if cache is not None and docid in cache:
                self.docs_data[docid] = cache[docid]
                continue
            else:
                self.update_cache.append('documents')
                p = subprocess.run(['galago', 'doc',
                                    '--index={}'.format(PATHS['galago_idx']),
                                    '--id=PMID-{}'.format(docid)],
                                   stdout=subprocess.PIPE)
                doc = p.stdout.decode('utf-8')
                # find the fields in the raw text
                fields = {'text': 'TEXT',
                          'title': 'TITLE',
                          'journal': 'MEDLINE_TA'}
                doc_ = {}
                for k, v in fields.items():
                    tag_len = len('<{}>'.format(v))
                    start = doc.find('<{}>'.format(v)) + tag_len
                    end = doc.find('</{}>'.format(v))
                    if start >= end or start <= tag_len or end <= 0:
                        contents = ''
                    else:
                        contents = doc[start:end]
                    doc_[k] = contents
                self.docs_data[docid] = doc_
                logger.info("reading document {} done".format(docid))

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
        col1 = '\n'.join(self.expected_docs)
        col2 = []  # Returned documents
        col3 = []  # Scores
        for docid in self.rankings_fusion[:topn]:
            docid_ = \
                colored(docid, 'blue') if docid in self.expected_docs else docid
            col2.append('{:>8}'.format(docid_))
            idx = self.rankings.index(docid)
            scores_ = "{:.2f}".format(self.scores['retrieval'][idx])
            # todo. add other kinds of scores (like reranking scores)
            if 'qasim' in self.scores:
                scores_ += "/{:.2f}".format(np.max(self.scores['qasim'][idx]))
            if 'journal' in self.scores:
                scores_ += "/{:.2f}".format(np.max(self.scores['journal'][idx]))
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
        # todo. compare with [merge_scores] in rerank_qaprox.py
        ndocs = len(self.scores['retrieval'])
        weights = {}
        for w in weights_str.split(','):
            k, v = w.split(':')
            if k == 'retrieval':
                weights['retrieval'] = float(v)
            if k == 'qasim':
                weights['qasim'] = float(v)
            if k == 'journal':
                weights['journal'] = float(v)

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
                fusion[i] += np.max(score) * weights['qasim']
        if 'journal' in weights and 'journal' in self.scores:
            fusion += np.array(self.scores['journal']) * weights['journal']
        self.scores['fusion'] = fusion.tolist()
        # Order by fusion scores
        self.rankings_fusion = \
            [docid for _, docid
             in sorted(zip(self.scores['fusion'], self.rankings),
                       key=lambda pair: pair[0], reverse=True)]

