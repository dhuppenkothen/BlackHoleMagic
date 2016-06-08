
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
import feature_engineering
import plotting

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def features_pca(datadir, tseg, log_features=None, ranking=None, axes=None,
                 alpha=0.8, palette="Set3", algorithm="pca"):

    features, labels, lc, \
    hr, tstart = feature_engineering.load_features(datadir, tseg,
                                                   log_features=log_features,
                                                   ranking=ranking)

    features_lb, labels_lb, lc_lb, \
    hr_lb, tstart_lb = feature_engineering.labelled_data(features, labels,
                                                         lc, hr, tstart)

    fscaled, fscaled_lb = feature_engineering.scale_features(features,
                                                             features_lb)

    fscaled_full = np.vstack([fscaled["train"], fscaled["val"],
                              fscaled["test"]])

    labels_all = np.hstack([labels["train"], labels["val"], labels["test"]])

    if algorithm == 'pca':
        pc = PCA(n_components=2)
        fscaled_trans = pc.fit(fscaled_full).transform(fscaled_full)
    elif algorithm == "tsne":
        fscaled_trans = TSNE(n_components=2).fit_transform(fscaled_full)

    sns.set_style("whitegrid")
    plt.rc("font", size=24, family="serif", serif="Computer Sans")
    plt.rc("axes", titlesize=20, labelsize=20)
    plt.rc("text", usetex=True)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    # make a Figure object
    if axes is None:
        fig, axes = plt.subplots(1,2,figsize=(16,6), sharey=True)

    ax1, ax2 = axes[0], axes[1]

    ax1 = plotting.scatter(fscaled_trans, labels_all, ax=ax1)

    # second panel: physical labels:
    labels_phys = feature_engineering.convert_labels_to_physical(labels)

    labels_all_phys = np.hstack([labels_phys["train"], labels_phys["val"],
                                 labels_phys["test"]])

    ax2 = plotting.scatter(fscaled_trans, labels_all_phys, ax=ax2)

    plt.tight_layout()

    return ax1, ax2
