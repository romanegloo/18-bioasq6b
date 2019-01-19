"""
Run evaluation and optimization using the saved scores

scores object structure
[
    {
        "qbody": question_str,
        "qid": qid,
        "ranked_docs": [d1, d2, ..., dk],
        "relevancy": [r1, r2, ..., rk],
        "scores": {
            "fusion": [ ... ],
            "journal": [ ... ],
            "qasim": [
                [ ... ],
                ...
            ],
            "retrieval": [ ... ],
            "semmeddb": [
                [exact_match, qmesh_match_ratio]
                ...
            ]
        }
    }
]

functions:
- On the given year and batch, return evaluation measures (recall, precision,
  f1, map, gmap)
- run random search algorithm
- run a L2R algorithms on the scores
- all the functions in interactive_train should be moved to this new script.
After then delete *interactive_train* file.
"""
import code
import glob
import os
import re
import json
import subprocess
import numpy as np
from numpy import inf
import random
from collections import OrderedDict
import logging
from apiclient.discovery import build
from httplib2 import Http
from oauth2client import file as oauth_file, client, tools


from BioAsq6B import PATHS
from BioAsq6B.common \
    import softmax, measure_performance, AverageMeter, AdaptiveRandomSearch

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

verbose = False

gt_data = dict()  # All the questions with GTs
scores = []  # Locally stored scores of retrieval and re-rankings
scores_batches = []  # Limit to specific years and batches
default_score_weights = OrderedDict([
    ('retrieval', 0.1716),
    ('qasim', 0.7066),
    ('journal', 0.0203),
    ('semmeddb1', 0.0069),
    ('semmeddb2', 0.0945)
])
l2r_model_name = [
    'MART', 'RankNet', 'RankBoost', 'AdaRank', 'CoordinateAscent',
    'n/a', 'LambdaMART', 'ListNet', 'RandomForests',
]

banner = """
=== Usage ===
  >>> usage()
      [prints out this banner]
      
  >>> eval(year=5, batches=[2, 3], weights={'retrieval': 1.0}, ranker=1)
      [With the given weights, evaluate the performance by score fusion]
      
  >>> run_ars(year=6)
      [Run adaptive random search for score fusion optimization]
      
  >>> run_l2r(year=5, ranker=1)
      [Run RankLib for optimization]
      
  >>> write_recall0_questions_glapi()
      [Get the retrieval results and write all the questions of recall score 0]
      
  >>> verbose = True
      [set verbose mode]
      
  >>> default_score_weights = {'retrieval': 0.5, 'qasim': 0.5}
      [reset the weights]
"""

debug = None
qasim_stats = None

# Related to GoogleAPIs
SCOPES = 'https://www.googleapis.com/auth/spreadsheets'
SAMPLE_SPREADSHEET_ID = '1HO1DpiHPxGzpUdIIuOlKS9VTpD4xDGD7EPhviP-hueI'


def usage():
    print(banner)
    print("=== Current score fusion weights ===")
    for k, v in default_score_weights.items():
        print("  - {}: {}".format(k, v))


def init():
    global qasim_stats

    print("=== Initializing ===")
    # Aggregate all the scores; filenames in "scores-[3-6]_[1-5].json" format
    print("  Reading all the saved scores...")
    qascores = []  # Place to store all the QASim scores for later use
    for f in os.listdir(PATHS['runs_dir']):
        m = re.search(r'^scores-([6])_([1-5])\.json', f)
        if m:
            scores_file = os.path.join(PATHS['runs_dir'], f)
            records = json.load(open(scores_file))
            # Handle -inf scores
            for rec in records:
                if 'qasim' in rec['scores']:
                    for x in rec['scores']['qasim']:
                        x[x == -inf] = 0
                        qascores.extend(x)
            scores.extend(records)
            scores_batches.append((int(m.group(1)), int(m.group(2))))
    print("  {} questions and the scores are read".format(len(scores)))
    # Reading example from the training dataset
    print("  Reading train dataset...")
    filepath = os.path.join(PATHS['train_dir'], 'BioASQ-trainingDataset6b.json')
    data = json.load(open(filepath))
    for q in data['questions']:
        gt_data[q['id']] = q
    # Reading all the test datasets; filenames in "phaseB_[1-6]_0[1-5].json"
    print("  Reading test datasets...")
    for f in os.listdir(PATHS['test_dir']):
        if re.match(r'^phaseB_[5-6]b_0[1-5]\.json', f):
            filepath = os.path.join(PATHS['test_dir'], f)
            data = json.load(open(filepath))
            for q in data['questions']:
                if q['id'] not in gt_data:
                    gt_data[q['id']] = q
    print("  {} examples read from the test datasets".format(len(gt_data)))
    na_qascores = np.array(qascores)
    qasim_stats = [na_qascores.min(), na_qascores.max(), na_qascores.mean(),
                   na_qascores.std()]
    print("  QASim stats: min ({:.4f}) max ({:.4f}) mean({:.4f}) std({:.4f})"
          ''.format(*qasim_stats))


def randomize_score_weights(keys=None):
    global default_score_weights
    all_features = ['retrieval', 'qasim', 'journal', 'semmeddb1', 'semmeddb2']
    if keys is None:
        keys = all_features
    else:
        if not all([k in all_features for k in keys]):
            print("key name should be one of {}".format(all_features))
            return
    a = np.random.random(len(keys))
    a /= a.sum()
    default_score_weights = OrderedDict.fromkeys(keys)
    for i, k in enumerate(keys):
        default_score_weights[k] = a[i]
    print("Random weights: {}".format(default_score_weights))


def merge_scores(rec, weights, stats):
    ndocs = len(rec['ranked_docs'])
    rel_docs = [link.split('/')[-1] for link
                in gt_data[rec['qid']]['documents']]
    # Get the metrics of the original ranked list by the retrieval scores
    prec, recall, f1, avg_prec = \
        measure_performance(rel_docs, rec['ranked_docs'])
    stats['avg_prec'].update(prec)
    stats['avg_recall'].update(recall)
    stats['avg_f1'].update(f1)
    stats['map0'].update(avg_prec)
    # Normalize retrieval scores
    scores_fusion = softmax(np.array(rec['scores']['retrieval'])) * \
                    weights['retrieval'] * ndocs
    if 'qasim' in weights and 'qasim' in rec['scores']:
        max_scores = [max(scores) for scores in rec['scores']['qasim']]
        # Normalize: min-max
        max_scores = [(s - qasim_stats[0])/(qasim_stats[1]-qasim_stats[0])
                      for s in max_scores]
        scores_fusion += np.array(max_scores) * weights['qasim']
    if 'journal' in weights and 'journal' in rec['scores']:
        j_scores = np.zeros((ndocs))
        j_scores[:len(rec['scores']['journal'])] = rec['scores']['journal']
        scores_fusion += j_scores * weights['journal']
    if 'semmeddb1' in weights and 'semmeddb2' in weights \
            and 'semmeddb' in rec['scores']:
        scores_fusion += np.array(rec['scores']['semmeddb']).dot(
            [weights['semmeddb1'], weights['semmeddb2']])
    y_pred = [docid for _, docid
              in sorted(zip(scores_fusion, rec['ranked_docs']),
                        key=lambda pair: pair[0], reverse=True)]
    _, _, _, avg_prec = measure_performance(rel_docs, y_pred)
    if verbose:
        print('qid: {} '
              '(prec. {:.4f}, recall {:.4f}, f1 {:.4f}, ap {:.4f})'
              ''.format(rec['qid'], prec, recall, f1, avg_prec))
    stats['map1'].update(avg_prec)


def eval(year, batches=None, weights=None, ranker=6):
    if batches is None:
        batches = [1, 2, 3, 4, 5]
    elif type(batches) != list:
        batches = [batches]
    if weights is None:
        weights = default_score_weights
    for b in batches:
        if (year, b) in scores_batches:
            print("Evaluating year {} batch {}...".format(year, b))
            stats = {
                'avg_prec': AverageMeter(),
                'avg_recall': AverageMeter(),
                'avg_f1': AverageMeter(),
                'map0': AverageMeter(),  # MAP by only retrieval score
                'map1': AverageMeter(),  # MAP by running score_fusion
                'map2': AverageMeter()   # MAP by running RankNet
            }
            for rec in scores:  # Score fusion by running ARS
                if rec['year'] == year and rec['batch'] == b:
                    merge_scores(rec, weights, stats)
            if ranker != 0:
                eval_l2r_batch(year, b, ranker, stats=stats)  # RankLib
            print('=== performance measures (year {} batch {}) ==='
                  ''.format(year, b))
            print(' - MAP_0 (sort by retrieval scores): {:.4f}'
                  ''.format(stats['map0'].avg))
            print(' - MAP_1 (sort by fusion scores -- ARS)): {:.4f}'
                  ''.format(stats['map1'].avg))
            if ranker != 0:
                print(' - MAP_2 (sort by RankLib[{}] scores): {:.4f}'
                      ''.format(l2r_model_name[ranker], stats['map2'].avg))
            print(' - Precision: {:.4f}'.format(stats['avg_prec'].avg))
            print(' - Recall: {:.4f}'.format(stats['avg_recall'].avg))
            print(' - F1: {:.4f}'.format(stats['avg_f1'].avg))


def eval_l2r_batch(year, batch, ranker, stats=None):
    """Run RankLib on the specified batch to get the scores and metrics"""
    if ranker == 5:  # not exist
        return
    # Run RankLib to get the score2
    test_file = os.path.join(PATHS['runs_dir'],
                             'l2r_test-{}_{}.txt'.format(year, batch))
    model_file = os.path.join(PATHS['runs_dir'], 'ranklib_models',
                              '{}_{}.mdl'.format(l2r_model_name[ranker], year))
    output_file = os.path.join(PATHS['runs_dir'],
                               'scores-{}-{}_{}.out'.format(
                                   l2r_model_name[ranker], year, batch))
    p = subprocess.run(['java', '-jar', PATHS['ranklib_bin'],
                        '-load', model_file, '-rank', test_file,
                        '-score', output_file],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode < 0:
        logger.error("Failed to run RankLib {}".format(p.stdout))
    else:
        logger.info("RankLib score is saved [{}]".format(output_file))

    # Read the output file, and get y_pred through sorting by the Ranknet scores
    l2r_scores = dict()
    try:
        with open(output_file) as f:
            for line in f:
                score = line.split()
                if score[0] not in l2r_scores:
                    l2r_scores[score[0]] = list()
                l2r_scores[score[0]].append(float(score[2]))
    except FileNotFoundError:
        logger.warning('WARNING. Score file does not exist. Please run '
                       '"run_l2r(...)" first.')
        return
    # Measure performance and update MAP of the given stats
    # if stats is not None and 'map2' in stats:
    assert (year, batch) in scores_batches
    for rec in scores:
        if rec['year'] == year and rec['batch'] == batch:
            k = rec['qid']
            rel_docs = [link.split('/')[-1] for link in gt_data[k]['documents']]
            y_pred = [docid for _, docid
                      in sorted(zip(l2r_scores[k], rec['ranked_docs']),
                                key=lambda pair: pair[0], reverse=True)]
            prec, recall, f1, avg_prec = measure_performance(rel_docs, y_pred)
            if verbose:
                print('qid: {} '
                      '(prec. {:.4f}, recall {:.4f}, f1 {:.4f}, ap {:.4f})'
                      ''.format(rec['qid'], prec, recall, f1, avg_prec))
            if stats is not None:
                stats['map2'].update(avg_prec)


def objective_fn(vec, dataset, weight_keys=None):
    stats = {
        'avg_prec': AverageMeter(),
        'avg_recall': AverageMeter(),
        'avg_f1': AverageMeter(),
        'map0': AverageMeter(),
        'map1': AverageMeter(),
        'map2': AverageMeter(),
    }
    for rec in dataset:
        weights = dict()
        for i, k in enumerate(weight_keys):
            weights[k] = vec[i]
        merge_scores(rec, weights, stats)
    return 1 - stats['map1'].avg


def run_ars(year=6, features=None):
    assert year in range(1, 7)
    """Run ARS (Adaptive Random Search) for tranining a model for score 
    fusion with the given set of features """
    validation_iter = 5
    random_search_iter = 300
    test_perc = .2
    printouts = []
    # Read the scores on the given year or before
    data = []
    for rec in scores:
        if rec['year'] == year:
            data.append(rec)
    try:
        if features is not None:
            randomize_score_weights(features)
        weights = default_score_weights
        for iter in range(validation_iter):
            random.seed()
            random.shuffle(scores)
            test_size = int(len(scores) * test_perc)
            test_data = scores[:test_size]
            train_data = scores[test_size:]
            optimizer = AdaptiveRandomSearch(
                [[0, 1]] * len(weights), objective_fn,
                init_vec=list(weights.values()), weight_keys=weights.keys(),
                data=train_data, max_iter=random_search_iter,
            )
            best = optimizer.search()
            printouts.append('best {} fold #{}'.format(best['vector'], iter))
            # Measure over test_data
            map_test = \
                1 - objective_fn(best['vector'], test_data, weights.keys())
            printouts.append('MAP over test_data: {}'.format(map_test))

    except KeyboardInterrupt:
        for line in printouts:
            print(line)
        return
    for line in printouts:
        print(line)


def run_l2r(year=None, ranker=1):
    """Run RankLib (https://goo.gl/rZNxMb) for tranining a model. It
    assumes that the software is installed properly, and the set of features
    with its weights are defined in *default_score_weights* """
    if ranker == 5:
        logger.warning('WARN. ranker 5 does not exist.')
        return
    # Helper function to transform scores into RankLib readable data entries
    def convert_scores2features(record):
        lines = []
        qid = record['qid']
        scores_ = record['scores']
        ret_norm = softmax(scores_['retrieval'])
        ndocs = len(record['ranked_docs'])
        for i, docid in enumerate(record['ranked_docs']):
            target = record['relevancy'][i]
            ft_scores = []
            if 'retrieval' in default_score_weights:
                if 'retrieval' in scores_:
                    ft_scores.append('1:{:.6f}'.format(ret_norm[i]))
                    # ft_scores.append('1:{}'.format(scores_['retrieval'][i]))
                else:
                    ft_scores.append('1:0.000000')
            if 'qasim' in default_score_weights:
                if 'qasim' in scores_:
                    ft_scores.append('2:{:.6f}'.format(
                        max(scores_['qasim'][i])))
                else:
                    ft_scores.append('2:0.000000')
            if 'journal' in default_score_weights:
                if 'journal' in scores_:
                    ft_scores.append('3:{:.6f}'.format(scores_['journal'][i]))
                else:
                    ft_scores.append('3:0.000000')
            if 'semmeddb1' in default_score_weights and \
                    'semmeddb2' in default_score_weights:
                if 'semmeddb' in scores_:
                    ft_scores.append('4:{:.6f} 5:{:.6f}'
                                     ''.format(scores_['semmeddb'][i][0],
                                               scores_['semmeddb'][i][1]))
                else:
                    ft_scores.append('4:0.000000 5:0.000000')
            comment = docid
            line = '{} qid:{} {} # docid={}'.format(
                target, qid, ' '.join(ft_scores), comment)
            lines.append(line)
        return lines

    # Use all the observations, if year is not given
    if year is None:
        year = 6
    data_train = []
    data_test = [[], [], [], [], []]
    # Read the scores on the given year or before, and transform the
    # records for ranklib runs such that,
    #   "3 qid:1 1:1 2:1 3:0 4:0.2 5:0 # 1A"
    #   <target> qid:<qid> <feature>:<value> ... <feature>:<value> # <comment>
    for rec in scores:
        if rec['year'] <= year:
            data_train.extend(convert_scores2features(rec))
        elif rec['year'] == year:
            b_idx = rec['batch'] - 1
            data_test[b_idx].extend(convert_scores2features(rec))

    # Save l2r data files for ranklib runs
    train_file = os.path.join(PATHS['runs_dir'], 'l2r_train.txt')
    if os.path.exists(train_file):
        os.remove(train_file)
    with open(train_file, 'w') as f:
        f.write('\n'.join(data_train))
    logger.info("Train dataset for RankLib runs is updated.")

    for f in glob.glob(os.path.join(PATHS['runs_dir'], 'l2r_test-*.txt')):
        if os.path.exists(f):
            os.remove(f)
    for i, b in enumerate(data_test):
        if len(b) > 0:
            test_file = os.path.join(PATHS['runs_dir'],
                                     'l2r_test-{}_{}.txt'.format(year, i+1))
            with open(test_file, 'w') as f:
                f.write('\n'.join(data_test[i]))
    logger.info("Test datasets for RankLib runs are updated.")

    # Run L2R rankers

    logger.info('Training a model with {} algorithm...'
                ''.format(l2r_model_name[ranker]))
    output_file = os.path.join(PATHS['runs_dir'], 'ranklib_models',
                               '{}_{}.mdl'.format(l2r_model_name[ranker], year))
    # Commands
    command = [
        'java', '-jar', PATHS['ranklib_bin'], '-train', train_file,
        '-tvs', '.8', '-metric2t', 'MAP', '-metric2T', 'MAP',
        '-save', output_file, '-ranker', str(ranker)
    ]
    # Extend ranker specific parameters to the command
    if ranker == 8:
        command.extend(['-rtype', '6'])

    p = subprocess.run(command, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    if p.returncode < 0:
        logger.error("Failed to run RankLib {}".format(p.stdout))
    else:
        logger.info("RankNet model is saved [{}]".format(output_file))


def write_recall0_questions_glapi():
    cred_json = '/home/jno236/research/ref/GoogleAPI/credentials.json'
    store = oauth_file.Storage('token.json')
    creds = store.get()

    # Build values
    values = list()
    for e in scores:
        if all(r==0 for r in e['relevancy']):
            gt = gt_data[e['qid']]
            docs = [d.split('/')[-1] for d in gt['documents']]
            answers = []
            if 'ideal_answer' in gt:
                for ans in gt['ideal_answer']:
                    answers.append(re.sub(r'\n', ', ', ans))
            values.append(['', gt['id'], gt['body'],
                           '\n'.join(answers), ', '.join(docs)])
    body = {
        'values': values
    }
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets(cred_json, SCOPES)
        creds = tools.run_flow(flow, store)
    service = build('sheets', 'v4', http=creds.authorize(Http()),
                    cache_discovery=False)

    result = service.spreadsheets().values().append(
        spreadsheetId=SAMPLE_SPREADSHEET_ID, range='RecallZeroQuestions',
        valueInputOption='RAW', body=body).execute()
    print('{0} cells appended.'.format(result \
                                       .get('updates') \
                                       .get('updatedCells')));


init()
code.interact(banner, local=locals())
