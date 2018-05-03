"""From the cached score, optimize the scoring weights using random search"""

"""
structure of 'cached_scores'
{
    [id:qid]: {
        'rankings': {d1, d2, ..., dm},
        'scores-ret': {r1, r2, ..., rm},
        'scores-qasim': {
            'docid1': [s1, s2, ..., sn},
            ...
        },
        'scores-journal': [j1, j2, ..., jm]
    }
}
"""
import code
import os
import pickle
import logging
import numpy as np
import random
import math
import json

from BioAsq6B import PATHS
from BioAsq6B.common \
    import AverageMeter, measure_performance, AdaptiveRandomSearch

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

banner = """
usage:
  >>> usage()
        [prints out this banner]
  >>> train(ft_list=['ret', 'qasim', 'journal', 'semmeddb1', 'semmeddb2']) or train()
        [run training loop of random search optimization]
  >>> eval({'ret': 0.5, 'qasim': 0.4, 'journal': 0.1}) or eval()
        [run test evaluation with the given parameters]
"""

data = None
scores = []


def init():
    global data
    # Read cached scores first
    file_scores = os.path.join(PATHS['data_dir'], 'cache-scores.pkl')
    data = pickle.load(open(file_scores, 'rb'))
    logger.info('{} questions are read'.format(len(data)))


def usage():
    print(banner)


def run_epoch(vec, dataset):
    assert len(scores) == len(vec)
    stats = {
        'avg_prec': AverageMeter(),
        'avg_recall': AverageMeter(),
        'avg_f1': AverageMeter(),
        'map': AverageMeter(),
        'logp': AverageMeter()
    }
    for docid, val in dataset.items():
        ndocs = len(val['scores-ret'])
        # Merge scores
        ret_norm = list(map(lambda x: math.exp(x), val['scores-ret']))
        ret_norm = [float(s)/sum(ret_norm) for s in ret_norm]
        fusion_ = None
        if 'ret' in scores:
            fusion_ = vec[scores.index('ret')] * ndocs * np.array(ret_norm)
        if 'qasim' in scores and 'scores-qasim' in val:
            # for i, score in enumerate(val['scores-qasim']):
            for i, docid in enumerate(val['rankings']):
                if docid in val['scores-qasim']:
                    fusion_[i] += np.max(val['scores-qasim'][docid]) *\
                                  vec[scores.index('qasim')]
        if 'journal' in scores and 'scores-journal' in val:
            fusion_ += np.array(val['scores-journal']) * \
                       vec[scores.index('journal')]
        if 'semmeddb1' in scores and 'scores-semmeddb' in val:
            fusion_ += np.array([v[0] for v in val['scores-semmeddb']]) *\
                       vec[scores.index('semmeddb1')]

        if 'semmeddb2' in scores and 'scores-semmeddb' in val:
            fusion_ += np.array([v[1] for v in val['scores-semmeddb']]) * \
                       vec[scores.index('semmeddb2')]
        scores_fusion = fusion_.tolist()
        rankings_fusion = [docid for _, docid
                           in sorted(zip(scores_fusion, val['rankings']),
                                     key=lambda pair: pair[0], reverse=True)]
        # Re-rank
        precision, recall, f1, avg_prec = \
            measure_performance(val['expected-docs'], rankings_fusion)
        stats['avg_prec'].update(precision)
        stats['avg_recall'].update(recall)
        stats['avg_f1'].update(f1)
        stats['map'].update(avg_prec)
        stats['logp'].update(math.log(precision + 1e-6))
    return stats


def objective_fn(vec, dataset):
    """With a list of random weights (vec), get scores and compute evaluation
    measures. Return the cost (loss)"""
    stats = run_epoch(vec, dataset)
    return 1 - stats['map'].avg


def train(ft_list=['ret', 'qasim', 'journal', 'semmeddb1', 'semmeddb2']):
    global scores
    scores = ft_list
    """random search results on dataset year 5:
        0.454043149 (ret) 0.435235034 (qasim) 0.110721817 (journal)"""
    # Repeated randomly sub-sampling validation; split the examples into
    # train and test group (test size: .2)
    validation_iter = 5
    random_search_iter = 200
    test_size = .2
    printouts = []
    try:
        for iter in range(validation_iter):
            test_keys = random.sample(list(data), int(len(data) * test_size))
            test_data = dict()
            train_data = dict()
            for k, v in data.items():
                if k in test_keys:
                    test_data[k] = v
                else:
                    train_data[k] = v
            init_weight = \
                np.random.dirichlet(np.ones(len(scores)), size=1).squeeze().tolist()
            optimizer = AdaptiveRandomSearch(
                [[0, 1]] * len(scores), objective_fn, init_weight,
                data=train_data, max_iter=random_search_iter
            )
            best = optimizer.search()
            printouts.append('fold#{}'.format(iter))
            printouts.append('BEST params: {}'.format(best['vector']))
            # Test
            stats = run_epoch(best['vector'], train_data)
            printouts.append('MAP: {}, vec: {}'
                             ''.format(stats['map'].avg, best['vector']))
    except KeyboardInterrupt:
        for line in printouts:
            logger.info(line)
    for line in printouts:
        logger.info(line)


def eval(params=None, year=5):
    """Given parameter dictionary, run evaluation to get the results.
     Note that all the scores are assumably cached in the file"""
    if params is None:
        params = {'ret': 0.4580, 'qasim': 0.0049, 'journal': 0.0724,
                 'semmeddb1': 0.2795, 'semmeddb2': 0.1852}
    global scores
    scores = []
    vec = []
    for k, v in params.items():
        scores.append(k)
        vec.append(v)
    # Read the question list of the given year by batches
    datasets = []
    batches = [1, 2, 3, 4, 5]
    for b in batches:
        data_b = {}
        batch_file = os.path.join(PATHS['test_dir'],
                                  'phaseB_{}b_0{}.json'.format(year, b))
        with open(batch_file) as f:
            testdata = json.load(f)
        for q in testdata['questions']:
            if q['id'] not in data:
                continue
            data_b[q['id']] = data[q['id']]
        datasets.append(data_b)
    # Run by batch
    for b in batches:
        stats = run_epoch(vec, datasets[b-1])
        report = ("(Test Run -- year{} batch{} #{}) "
                  "prec.: {:.4f}, recall: {:.4f}, f1: {:.4f} map: {:.4f}"
                  ).format(year, b, len(datasets[b-1]),
                           stats['avg_prec'].avg,
                           stats['avg_recall'].avg,
                           stats['avg_f1'].avg,
                           stats['map'].avg)
        logger.info(report)

init()
code.interact(banner, local=locals())
