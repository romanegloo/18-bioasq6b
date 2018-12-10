import os
import json
import unicodedata
from BioAsq6B import PATHS

questions = []
# Read from training dataset year 6
dataset6_file = os.path.join(PATHS['train_dir'],
                             'BioASQ-trainingDataset6b.json')
data = json.load(open(dataset6_file))
for q in data['questions']:
    questions.append((q['id'], q['body']))

for b in range(1, 6):
    testset_file = os.path.join(PATHS['test_dir'],
                                "phaseB_6b_0{}.json".format(b))
    data = json.load(open(testset_file))
    for q in data['questions']:
        questions.append((q['id'], q['body']))

for q in questions:
    id = q[0]
    text = unicodedata.normalize('NFKD', q[1]).encode('ascii', 'ignore')
    print('{}|{}'.format(id, text))
