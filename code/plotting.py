import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import seaborn as sns

import numpy as np

import sklearn.metrics

from collections import Counter


def _compute_trans_matrix(labels):
    unique_labels = np.unique(labels)
    nlabels = len(unique_labels)

    labels_numerical = np.array([np.where(unique_labels == l)[0][0] \
                                 for l in labels])
    labels_numerical = labels_numerical.flatten()

    transmat = np.zeros((nlabels,nlabels))
    for (x,y), c in Counter(zip(labels_numerical,
                                labels_numerical[1:])).iteritems():
        transmat[x,y] = c

    transmat_p = np.zeros_like(transmat)
    for i,t in enumerate(transmat):
        transmat_p[i,:] = t/np.max(t)

    return unique_labels, transmat, transmat_p



def transition_matrix(labels, ax=None):

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(9,9))

    unique_labels, transmat, transmat_p = _compute_trans_matrix(labels)

    sns.set_style("whitegrid")
    plt.rc("font", size=24, family="serif", serif="Computer Sans")
    plt.rc("axes", titlesize=20, labelsize=20)
    plt.rc("text", usetex=True)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    ax.pcolormesh(np.log10(transmat), cmap=cmap.viridis)
    ax.set_ylabel('Initial state')
    ax.set_xlabel('Final state')
    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels(unique_labels, rotation=70)
    ax.set_yticks(range(len(unique_labels)))
    ax.set_yticklabels(unique_labels)

    return ax


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
    """
    Make a scatter plot of dimensions 0 and 1 in features, with scatter
    points coloured by labels.

    Parameters
    ----------
    features : matrix (N, M)
        A (N, M) matrix of `features` with N samples and M features for each
        sample.

    labels : iterable
        A list or array of N labels corresponding to the N feature vectors
        in `features`.

    ax : matplotlib.Axes object
        The Axes object to plot into

    palette : str
        The string of the color palette to use for the different classes
        By default, "Set3" is used.

    alpha : {0,1} float
        Float between 0 and 1 controlling the transparency of the scatter
        points.

    Returns
    -------
    ax : matplotlib.Axes
        The Axes object with the plot.

    """
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
                   color="grey", alpha=alpha, label="unclassified")

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


import powerspectrum

def plot_misclassifieds(features, trained_labels, real_labels, lc_all, hr_all,
                        nexamples=6, namestr="misclassified", datadir="./"):

    """
    Find all mis-classified light curves and plot them with examples of the
    real and false classes.

    Parameters
    ----------
    features : numpy.ndarray
        The (N,M) array with N samples (rows) and M features (columns) per
        sample

    trained_labels : iterable
        The list or array with the trained labels

    real_labels : iterable
        The list or array with the true labels

    lc_all : list
        A list of N light curves corresponding to each sample

    hr_all : list
        A list of N hardness ratio measurements corresponding to each sample

    nexamples : int
        The number of examples to plot; default is 6

    namestr : str
        The string to append to each plot for saving to disc,
        default: "misclassified"

    datadir : str
        The path of the directory to save the figures in


    """
    misclassifieds = []
    for i,(f, lpredict, ltrue, lc, hr) in enumerate(zip(features,
                                                        trained_labels,
                                                        real_labels, lc_all,
                                                        hr_all)):
        if lpredict == ltrue:
            continue
        else:
            misclassifieds.append([f, lpredict, ltrue, lc, hr])

    for j,m in enumerate(misclassifieds):
        pos_human = np.random.choice([0,3], p=[0.5, 0.5])
        pos_robot = int(3. - pos_human)

        f = m[0]
        lpredict = m[1]
        ltrue = m[2]
        times = m[3][0]
        counts = m[3][1]
        hr1 = m[4][0]
        hr2 = m[4][1]
        print("Predicted class is: " + str(lpredict))
        print("Human classified class is: " + str(ltrue))
        robot_all = [[lp, lt, lc, hr] for lp, lt, lc, hr in \
                     zip(real_labels, trained_labels, lc_all, hr_all)\
                     if lt == lpredict ]
        human_all = [[lp, lt, lc, hr] for lp, lt, lc, hr in \
                     zip(real_labels, trained_labels, lc_all, hr_all)\
                     if lt == ltrue ]

        np.random.shuffle(robot_all)
        np.random.shuffle(human_all)
        robot_all = robot_all[:6]
        human_all = human_all[:6]

        sns.set_style("darkgrid")
        current_palette = sns.color_palette()
        fig = plt.figure(figsize=(10,15))

        def plot_lcs(times, counts, hr1, hr2, xcoords, ycoords,
                     colspan, rowspan):
            #print("plotting in grid point " + str((xcoords[0], ycoords[0])))
            ax = plt.subplot2grid((9,6),(xcoords[0], ycoords[0]),
                                  colspan=colspan, rowspan=rowspan)
            ax.plot(times, counts, lw=2, linestyle="steps-mid", rasterized=True)
            ax.set_xlim([times[0], times[-1]])
            ax.set_ylim([0.0, 12000.0])
            #print("plotting in grid point " + str((xcoords[1], ycoords[1])))

            ax = plt.subplot2grid((9,6),(xcoords[1], ycoords[1]),
                                  colspan=colspan, rowspan=rowspan)
            ax.scatter(hr1, hr2, facecolor=current_palette[1],
                       edgecolor="none", rasterized=True)
            ax.set_xlim([.27, 0.85])
            ax.set_ylim([0.04, 0.7])

            #print("plotting in grid point " + str((xcoords[2], ycoords[2])))
            ax = plt.subplot2grid((9,6),(xcoords[2], ycoords[2]),
                                  colspan=colspan, rowspan=rowspan)
            dt = np.min(np.diff(times))
            ps = powerspectrum.PowerSpectrum(times, counts=counts/dt,
                                             norm="rms")
            ax.loglog(ps.freq[1:], ps.ps[1:], linestyle="steps-mid",
                      rasterized=True)
            ax.set_xlim([ps.freq[1], ps.freq[-1]])
            ax.set_ylim([1.e-6, 10.])

            return

        ## first plot misclassified:
        plot_lcs(times, counts, hr1, hr2, [0,0,0], [0,2,4], 2, 2)

        ## now plot examples
        for i in range(4):
            r = robot_all[i]
            h = human_all[i]

            plot_lcs(h[2][0], h[2][1], h[3][0], h[3][1], [i+2, i+2, i+2],
                     [pos_human, pos_human+1, pos_human+2], 1, 1)
            plot_lcs(r[2][0], r[2][1], r[3][0], r[3][1], [i+2, i+2, i+2],
                     [pos_robot, pos_robot+1, pos_robot+2], 1, 1)

        ax = plt.subplot2grid((9,6),(8,pos_human+1))
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_xlabel("Human: %s"%ltrue, fontsize=20)
        ax = plt.subplot2grid((9,6),(8,pos_robot+1))
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_xlabel("Robot: %s"%lpredict, fontsize=20)
        plt.savefig(datadir,"misclassified%i.pdf"%j, format="pdf")
        plt.close()

    return