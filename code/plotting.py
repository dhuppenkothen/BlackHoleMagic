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
