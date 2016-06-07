
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import feature_engineering
from sklearn.decomposition import PCA


def features_pca(datadir, tseg, log_features=None, ranking=None, ax=None,
                 alpha=0.8, palette="Set3"):

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

    pc = PCA(n_components=2)
    fscaled_pca = pc.fit(fscaled_full).transform(fscaled_full)


    sns.set_style("whitegrid")
    plt.rc("font", size=24, family="serif", serif="Computer Sans")
    plt.rc("axes", titlesize=20, labelsize=20)
    plt.rc("text", usetex=True)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    unique_labels = np.unique(labels_all)
    unique_labels = np.delete(unique_labels,
                              np.where(unique_labels == "None")[0])

    # make a Figure object
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(12,9))

    # first plot the unclassified examples:
    ax.scatter(fscaled_pca[labels_all == "None",0],
               fscaled_pca[labels_all == "None",1],
               color="grey", alpha=alpha)

    # now make a color palette:
    current_palette = sns.color_palette(palette, len(unique_labels))

    for l, c in zip(unique_labels, current_palette):
        ax.scatter(fscaled_pca[labels_all == l,0],
                   fscaled_pca[labels_all == l,1], s=40,
                   color=c, alpha=alpha, label=l)

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_xlim(-6.2, 8.0)
    ax.set_ylim(-7.0, 8.0)
    ax.legend(loc="upper right", prop={"size":14})

    plt.tight_layout()
    #plt.savefig(datadir+"grs1915_features_pca.pdf", format="pdf")

    return ax
