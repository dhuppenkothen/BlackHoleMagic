
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
import feature_engineering
import plotting

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier


try:
    import cPickle as pickle
except ImportError:
    import pickle


def plot_scores(datadir, scores):
    max_scores = []
    for s in scores:
        max_scores.append(np.max(s))

    sns.set_style("whitegrid")
    plt.rc("font", size=24, family="serif", serif="Computer Sans")
    plt.rc("axes", titlesize=20, labelsize=20)
    plt.rc("text", usetex=True)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    fig, ax = plt.subplots(1,1,figsize=(9,7))

    ax.scatter(np.arange(len(max_scores)), max_scores, marker="o",
               c=sns.color_palette()[0], s=40)
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Validation accuracy")

    plt.savefig(datadir+"grs1915_greedysearch_scores.pdf", format="pdf")
    plt.close()


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



def all_figures():
    datadir = "../../"

    # read in the results from the greedy feature engineering
    with open(datadir+"grs1915_greedysearch_res.dat" ,'r') as f:
        data = pickle.load(f)

    scores = data["scores"]
    ranking = data["ranking"]

    # Plot the maximum validation score for each feature in the greedy search
    plot_scores(datadir, scores)

    # First the comparison between Belloni classification + other classification
    # using PCA
    ax1, ax2 = features_pca(datadir, 1024., log_features=None, ranking=ranking,
                            axes=None, alpha=0.8, palette="Set3")

    ax2.set_ylabel("")
    plt.tight_layout()
    plt.savefig(datadir+"grs1915_features_pca.pdf", format="pdf")
    plt.close()

    # Same as before, using t-SNE
    ax1, ax2 = features_pca(datadir, 1024., log_features=None, ranking=ranking,
                            axes=None, alpha=0.8, palette="Set3",
                            algorithm="tsne")

    ax2.set_ylabel("")
    plt.tight_layout()
    plt.savefig(datadir+"grs1915_features_tsne.pdf", format="pdf")
    plt.close()

    # read in features + related information
    tseg = 1024.
    log_features = log_features = [2, 5, 6, 7, 9, 10, 11, 14, 16]

    features, labels, lc, \
    hr, tstart = feature_engineering.load_features(datadir, tseg,
                                                   log_features=log_features,
                                                   ranking=ranking)

    # get out the classified features and labels only
    features_lb, labels_lb, lc_lb, \
    hr_lb, tstart_lb = feature_engineering.labelled_data(features, labels,
                                                         lc, hr, tstart)

    # scale features using StandardScaler
    fscaled, fscaled_lb = feature_engineering.scale_features(features,
                                                             features_lb)

    # full set of scaled features
    fscaled_full = np.vstack([fscaled["train"], fscaled["val"],
                              fscaled["test"]])

    # all labels in one array
    labels_all = np.hstack([labels["train"], labels["val"], labels["test"]])

    # get classified features + labels
    fscaled_train = fscaled_lb["train"]
    fscaled_test = fscaled_lb["test"]
    fscaled_val = fscaled_lb["val"]

    labels_train = labels_lb["train"]
    labels_test = labels_lb["test"]
    labels_val = labels_lb["val"]

    # Do RF classification
    max_depth=50
    rfc = RandomForestClassifier(n_estimators=500, max_depth=max_depth)
    rfc.fit(fscaled_train, labels_train)

    print("Training score: " + str(rfc.score(fscaled_train, labels_train)))
    print("Validation score: " + str(rfc.score(fscaled_val, labels_val)))

    lpredict_val = rfc.predict(fscaled_val)
    lpredict_test = rfc.predict(fscaled_test)

    print("Test score: " + str(rfc.score(fscaled_test, labels_test)))

    # plot the confusion matrix
    fig, ax = plt.subplots(1,1,figsize=(9,9))
    ax = plotting.confusion_matrix(labels_val, lpredict_val, log=True, ax=ax)
    fig.subplots_adjust(bottom=0.15, left=0.15)
    plt.tight_layout()
    plt.savefig(datadir+"grs1915_supervised_cm.pdf", format="pdf")
    plt.close()


