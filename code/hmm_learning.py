
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

import seaborn as sns
sns.set_context("notebook", font_scale=2.5, rc={"axes.labelsize": 26})
plt.rc("font", size=24, family="serif", serif="Computer Sans")
plt.rc("axes", titlesize=20, labelsize=20)
plt.rc("text", usetex=True)

import numpy as np
import cPickle as pickle
import generaltools as gt

from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import matplotlib.cm as cmap
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

from grs1915_utils import state_distribution, state_time_evolution, plot_classified_lightcurves

def load_data(seg_length_unsupervised=256., datadir="../../"):
    """
    Load the GRS 1915+105 data in a useful format. Assumes the data
    comes in files that are output from the feature extraction step described
    elsewhere in this repository.

    Parameters
    -----------
    seg_length_unsupervised: float, optional, default 256
        The duration of the segments used for feature extraction.
        Because this is an unsupervised method, we'll use the shorter segments
        by default.

    datadir: string, optional, default "../../"
        The directory where the data is located, and where output will be saved


    Returns
    -------
    fscaled_train_sorted: np.ndarray
        The sorted array of training features; has dimensions (nsamples, nfeatures+1)
        The first column contains start times of each segment

    fscaled_test_sorted: np.ndarray
        The sorted array of test features; has dimensions (nsamples, nfeatures+1)
        The first column contains start times of each segment

    labels_train_sorted: list of strings
        The human-classified labels, just in case they might come in handy

    labels_test_sorted: list of strings
        The human-classified labels, just in case they might come in handy

    """


    features_train_full = np.loadtxt(datadir+"grs1915_%is_features_train.txt"%seg_length_unsupervised)
    features_test_full = np.loadtxt(datadir+"grs1915_%is_features_test.txt"%seg_length_unsupervised)
    features_val_full = np.loadtxt(datadir+"grs1915_%is_features_val.txt"%seg_length_unsupervised)

    labels_test_full = np.array(gt.conversion(datadir+"grs1915_%is_labels_test.txt"%seg_length_unsupervised)[0])
    labels_train_full = np.array(gt.conversion(datadir+"grs1915_%is_labels_train.txt"%seg_length_unsupervised)[0])
    labels_val_full = np.array(gt.conversion(datadir+"grs1915_%is_labels_val.txt"%seg_length_unsupervised)[0])

    tstart_train_full = np.loadtxt(datadir+"grs1915_%is_tstart_train.txt"%seg_length_unsupervised)
    tstart_test_full = np.loadtxt(datadir+"grs1915_%is_tstart_test.txt"%seg_length_unsupervised)
    tstart_val_full = np.loadtxt(datadir+"grs1915_%is_tstart_val.txt"%seg_length_unsupervised)

    feature_ranking = [10, 16, 8, 1, 26, 24, 9, 7, 4, 22, 28, 11,
                       20, 2, 15, 5, 12, 30, 18, 0, 27, 25, 6, 19,
                       3, 23, 17, 13, 29, 21, 14]

    ## we'll be using the first 20 re-ranked features
    max_features = 20

    """
    ## make new empty arrays for the ranked features
    features_new_train = np.zeros_like(features_train_full[:,:max_features])
    features_new_val = np.zeros_like(features_val_full[:,:max_features])
    features_new_test = np.zeros_like(features_test_full[:,:max_features])
    
    for i,f in enumerate(feature_ranking[:max_features]):
        if i in [0,2,3,6,7,11,13,15,16,19,20]:
            print("Making a log of parameter %i"%i)
            features_new_train[:,i] = np.log(features_train_full[:,f])
            features_new_val[:,i] = np.log(features_val_full[:,f])
            features_new_test[:,i] = np.log(features_test_full[:,f])
        else:
            features_new_train[:,i] = features_train_full[:,f]
            features_new_val[:,i] = features_val_full[:,f]
            features_new_test[:,i] = features_test_full[:,f]

    """
    features_new_train = features_train_full
    features_new_val = features_val_full
    features_new_test = features_test_full

    f_all = np.vstack((features_new_train, features_new_val, features_new_test))

    scaler_train = StandardScaler().fit(f_all)
    fscaled_train = scaler_train.transform(features_new_train)
    fscaled_val = scaler_train.transform(features_new_val)
    fscaled_test = scaler_train.transform(features_new_test)


    fscaled_train = np.vstack((fscaled_train, fscaled_val))
    labels_train = np.hstack((labels_train_full, labels_val_full))
    labels_test = labels_test_full

    tstart_train = np.hstack((tstart_train_full, tstart_val_full))
    tstart_test = tstart_test_full

    fscaled_train = np.vstack((tstart_train, fscaled_train.T)).T
    fscaled_test = np.vstack((tstart_test, fscaled_test.T)).T

    fscaled_train_sorted = fscaled_train[fscaled_train[:,0].argsort()]
    fscaled_test_sorted = fscaled_test[fscaled_test[:,0].argsort()]

    labels_train_sorted = labels_train[fscaled_train[:,0].argsort()]
    labels_test_sorted = labels_test[fscaled_test[:,0].argsort()]

    return fscaled_train_sorted, fscaled_test_sorted, labels_train_sorted, labels_test_sorted


def run_hmm(max_comp=5, n_cv=10, datadir="./"):
    """
    Run a Hidden Markov Model on the data.
    Saves a dictionary to disk with the following keywords:
        cv_scores: an array with dimensions (max_comp-1, n_cv) with the
                    cross validation scores
        cv_means: the means of of the CV scores for each number of states
        cv_std: standard deviations of the CV scores for each number of states
        test_score: the scores on the test set for each number of states

    Parameters
    ----------
    max_comp: int, optional, default 5
        The maximum number of states in the HMM

    n_cv: int, optional, default 10
        The number of cross-validation folds to use

    datadir: str, optional, default "./"
        The directory with the data, where the output will be saved


    """

    ftrain, ftest, ltrain, ltest = load_data(seg_length_unsupervised=16.)

    n_components = range(2,max_comp,1)
    ## run for up to 30 clusters
    cv_means, cv_std = [], []
    cv_scores_all = []
    test_score = []

    for n in n_components:
        print("Cross validation for classification with %i states \n"
              "--------------------------------------------------------"%n)
        ## make the samples for 10-fold cross-validation
        kf = cross_validation.KFold(ftrain.shape[0], n_folds=n_cv, shuffle=False)
        scores = []
        ## run through all samples
        for train, test in kf:
            X_train, X_test = ftrain[train, 1:], ftrain[test, 1:]
            y_train, y_test = ltrain[train], ltrain[test]

            model2 = hmm.GaussianHMM(n_components=n, covariance_type='full')
            model2.fit([X_train])
            pred_labels = model2.predict(X_train)

            ## get out those labels that are actually human classified
            classified_ind = np.where(y_train != "None")[0]
            #print(len(classified_ind))

            scores.append(model2.score(X_test))
            print("Score: " + str(model2.score(X_test)))
        cv_scores = np.array(scores)
        cv_means.append(np.mean(scores))
        cv_std.append(np.std(scores))

        model2 = hmm.GaussianHMM(n_components=n, covariance_type="full")
        model2.fit([ftrain])
        test_score.append(model2.score(ftest))

        cv_scores_all.append(cv_scores)
        #si_means.append(np.mean(si_scores))
        #si_std.append(np.mean(si_scores))
        ## print mean and standard deviation of the 10 cross-validated scores
        print("Cross-validation score is %.2f +- %.4f. \n"
              "============================================\n"%(np.mean(cv_scores), np.std(cv_scores)))

        print("Test score is %.2f. \n"
              "============================\n"%model2.score(ftest))


    hmm_results = {"cv_scores":cv_scores_all, "cv_means":cv_means,
                   "cv_std":cv_std, "test_score":test_score}

    f = open(datadir+"grs1915_hmm_results.dat", "w")
    pickle.dump(hmm_results, f)
    f.close()

    return


run_hmm(31, n_cv=5, datadir="../../")
