import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import seaborn as sns

import numpy as np

import sklearn.metrics


def confusion_matrix(labels_true, labels_pred, log=False,
                     ax=None, cm=cmap.viridis):

    """
    Plot a confusion matrix between true and predicted labels
    from a machine learning classification task.

    Parameters
    ----------
    labels_true : iterable
        List or array with true labels

    labels_pred : iterable
        List or array with predicted labels

    log : bool
        Plot original confusion matrix or the log of the confusion matrix?
        Default is False

    ax : matplotlib.Axes object
        An axes object to plot into

    cm : matplotlib.colormap
        A matplotlib colour map

    Returns
    -------
    ax : matplotlib.Axes object
        The Axes object with the plot

    """

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(9,6))

    unique_labels = np.unique(labels_true)
    confmatrix = sklearn.metrics.confusion_matrix(labels_true, labels_pred, labels=unique_labels)

    if log:
        if np.any(confmatrix == 0):
            confmatrix = confmatrix + np.min(confmatrix[np.nonzero(confmatrix)])/10.0

        confmatrix = np.log(confmatrix)

    sns.set_style("whitegrid")
    plt.rc("font", size=24, family="serif", serif="Computer Sans")
    plt.rc("axes", titlesize=20, labelsize=20)
    plt.rc("text", usetex=True)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    ax.pcolormesh(confmatrix, cmap=cm)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels(unique_labels, rotation=70)

    ax.set_yticks(range(len(unique_labels)))
    ax.set_yticklabels(unique_labels)

    return ax


def scatter(features, labels, ax=None, palette="Set3", alpha=0.8):

    sns.set_style("whitegrid")
    plt.rc("font", size=24, family="serif", serif="Computer Sans")
    plt.rc("axes", titlesize=20, labelsize=20)
    plt.rc("text", usetex=True)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    unique_labels = np.unique(labels)

    # make a Figure object
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(9,6), sharey=True)

    xlim = [np.min(features[:,0])-0.5, np.max(features[:,0])+3.5]
    ylim = [np.min(features[:,1])-0.5, np.max(features[:,1])+0.5]

    # If None is in labels, delete from set of unique labels and
    # plot all samples with label None in grey
    if "None" in unique_labels:
        unique_labels = np.delete(unique_labels,
                                  np.where(unique_labels == "None")[0])

        # first plot the unclassified examples:
        ax.scatter(features[labels == "None",0],
                   features[labels == "None",1],
                   color="grey", alpha=alpha)

    # now make a color palette:
    current_palette = sns.color_palette(palette, len(unique_labels))

    for l, c in zip(unique_labels, current_palette):
        ax.scatter(features[labels == l,0],
                   features[labels == l,1], s=40,
                   color=c, alpha=alpha, label=l)

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(loc="upper right", prop={"size":14})

    return ax