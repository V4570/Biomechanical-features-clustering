import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import manifold
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import os
import time

style.use('ggplot')

script_dir = os.path.dirname(__file__)
rel_file_path = "biomechanical/column_3C_weka.csv"

dataframe = pd.read_csv(os.path.join(script_dir, rel_file_path))

X = dataframe.loc[:, dataframe.columns != 'class'].values.astype('float')
Y = dataframe['class'].replace({'Normal': 0, 'Hernia': 1, 'Spondylolisthesis': 2})

n_samples, num_features = X.shape
centers = [[1, 1], [-1, -1], [1, -1]]


def embedding_plot(X, title):
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(X[:, 0], X[:, 1], lw=0, s=40, c=Y/10.)

    plt.xticks([]), plt.yticks([])
    plt.title(title)


t_tsne1 = time.time()
X_tsne = manifold.TSNE(n_components=2, init='pca').fit_transform(X)
t_tsne2 = time.time() - t_tsne1
print('Time to calculate t-SNE: {}'.format(t_tsne2))
embedding_plot(X_tsne, "t-SNE")


##############################################################################
# Compute clustering with Means
k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)

t0 = time.time()
k_means.fit(X_tsne)
t_batch = time.time() - t0
print('Time to calculate K-means: {}'.format(t_batch))
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels_unique = np.unique(k_means_labels)

##############################################################################
# Compute clustering with MiniBatchKMeans
mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=310,
                      n_init=10, max_no_improvement=10, verbose=0)
t0 = time.time()
mbk.fit(X_tsne)
t_mini_batch = time.time() - t0
mbk_means_labels = mbk.labels_
mbk_means_cluster_centers = mbk.cluster_centers_
mbk_means_labels_unique = np.unique(mbk_means_labels)


##############################################################################
# Plot result

fig = plt.figure(figsize=(8, 3))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']

# We want to have the same colors for the same cluster from the
# MiniBatchKMeans and the KMeans algorithm.

distance = euclidean_distances(k_means_cluster_centers,
                               mbk_means_cluster_centers,
                               squared=True)
order = distance.argmin(axis=1)

# KMeans
ax = fig.add_subplot(1, 3, 1)
for k, col in zip(range(len(centers)), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(
        cluster_center[0],
        cluster_center[1],
        'o',
        markerfacecolor=col,
        markeredgecolor='k',
        markersize=6
    )
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8,  'train time: %.2fs\ninertia: %f' % (
    t_batch, k_means.inertia_))

# MiniBatchKMeans
ax = fig.add_subplot(1, 3, 2)
for k, col in zip(range(len(centers)), colors):
    my_members = mbk_means_labels == k
    cluster_center = mbk_means_cluster_centers[order[k]]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(
        cluster_center[0],
        cluster_center[1],
        'o',
        markerfacecolor=col,
        markeredgecolor='k',
        markersize=6
    )
ax.set_title('MiniBatchKMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' % (t_mini_batch, mbk.inertia_))

# Initialise the different array to all False
different = (mbk_means_labels == 4)
ax = fig.add_subplot(1, 3, 3)

for l in range(len(centers)):
    different += ((k_means_labels == k) != (mbk_means_labels == order[k]))

identic = np.logical_not(different)
ax.plot(X[identic, 0], X[identic, 1], 'w',
        markerfacecolor='#bbbbbb', marker='.')
ax.plot(X[different, 0], X[different, 1], 'w',
        markerfacecolor='m', marker='.')
ax.set_title('Difference')
ax.set_xticks(())
ax.set_yticks(())

plt.show()
