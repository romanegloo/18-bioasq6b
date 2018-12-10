import pickle
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

datafile = '../../../data/journal_mesh_dist_6.pkl'
data = pickle.load(open(datafile, 'rb'))


# Select random 20 journals and get the mesh terms of them
journals = [
    'Neurobiol Aging',
    'Evid Based Dent',
    'Am J Bot',
    'Viruses',
    'J Obes',
    'Acad Radiol',
    'Kyobu Geka',
    'Mol Inform',
    'J Sports Sci',
    'J Bone Metab',
    'Breast Cancer Res'
]
# journals = random.sample(data['journals'].keys(), 10)
table = np.zeros((data['dist_table'].shape[0], len(journals)))

# Filter data_table by the journals
for i, j in enumerate(journals):
    mesh = list(map(lambda x: data['jmesh2idx'][x],
                    data['journals'][j]['mesh']))
    table[:,i] = np.mean(data['dist_table'][:, mesh], axis=1)

cnt = np.sum(table, axis=1)
mask = 5 < cnt
qmesh = []
for i, m in enumerate(mask):
    if m:
        qmesh.append(data['idx2qmesh'][i])
table = np.log(table[mask] + 1)
df = pd.DataFrame(table, index=qmesh, columns=[name[:20] for name in journals])
g = sns.clustermap(df, yticklabels=True)
g.ax_heatmap.tick_params(labelsize=8)
g.ax_heatmap.set_ylabel('MeSH terms in Questions', fontsize=10)
plt.show()

