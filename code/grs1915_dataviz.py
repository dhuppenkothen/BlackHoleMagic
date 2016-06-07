
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import feature_engineering
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

    unique_labels = np.unique(labels_all)
    unique_labels = np.delete(unique_labels,
                              np.where(unique_labels == "None")[0])

    # make a Figure object
    if axes is None:
        fig, axes = plt.subplots(1,2,figsize=(16,6), sharey=True)

    xlim = [np.min(fscaled_trans[:,0])-0.5, np.max(fscaled_trans[:,0])+0.5] # [-6.2, 8.0]
    ylim = [np.min(fscaled_trans[:,1])-0.5, np.max(fscaled_trans[:,1])+0.5] # [-7.0, 8.0]
    ax1, ax2 = axes[0], axes[1]

    # first plot the unclassified examples:
    ax1.scatter(fscaled_trans[labels_all == "None",0],
               fscaled_trans[labels_all == "None",1],
               color="grey", alpha=alpha)

    # now make a color palette:
    current_palette = sns.color_palette(palette, len(unique_labels))

    for l, c in zip(unique_labels, current_palette):
        ax1.scatter(fscaled_trans[labels_all == l,0],
                   fscaled_trans[labels_all == l,1], s=40,
                   color=c, alpha=alpha, label=l)

    ax1.set_xlabel("PCA Component 1")
    ax1.set_ylabel("PCA Component 2")
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.legend(loc="upper right", prop={"size":14})

    # second panel: physical labels:
    labels_phys = feature_engineering.convert_labels_to_physical(labels)

    labels_all_phys = np.hstack([labels_phys["train"], labels_phys["val"],
                                 labels_phys["test"]])

    labels_unique_phys = np.unique(labels_all_phys)
    none_ind = np.where(labels_unique_phys == "None")[0]
    labels_unique_phys = np.delete(labels_unique_phys, none_ind)
    print("physical labels: " + str(labels_unique_phys))

    # first plot the unclassified examples:
    ax2.scatter(fscaled_trans[labels_all_phys == "None",0],
                fscaled_trans[labels_all_phys == "None",1],
                color="grey", alpha=alpha)

    # now make a color palette:
    current_palette = sns.color_palette(palette, len(labels_unique_phys))

    for l, c in zip(labels_unique_phys, current_palette):
        ax2.scatter(fscaled_trans[labels_all_phys == l,0],
                    fscaled_trans[labels_all_phys == l,1], s=40,
                    color=c, alpha=alpha, label=l)

    ax2.set_xlabel("PCA Component 1")
    #ax2.set_ylabel("PCA Component 2")
    ax2.set_xlim(xlim)
    #ax2.set_ylim(ylim)
    ax2.legend(loc="upper right", prop={"size":14})

    plt.tight_layout()

    return ax1, ax2
