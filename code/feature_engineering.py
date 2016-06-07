import numpy as np
import generaltools as gt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import matplotlib.cm as cmap

import pickle


def transform_chi(labels):
    labels[labels == "chi1"] = "chi"
    labels[labels == "chi2"] = "chi"
    labels[labels == "chi3"] = "chi"
    labels[labels == "chi4"] = "chi"
    return labels


def load_features(datadir, tseg, log_features=None):

    features_train_full = np.loadtxt(datadir+"grs1915_%is_features_train.txt"%tseg)
    features_test_full = np.loadtxt(datadir+"grs1915_%is_features_test.txt"%tseg)
    features_val_full = np.loadtxt(datadir+"grs1915_%is_features_val.txt"%tseg)

    labels_test_full = np.array(gt.conversion(datadir+"grs1915_%is_labels_test.txt"%tseg)[0])
    labels_train_full = np.array(gt.conversion(datadir+"grs1915_%is_labels_train.txt"%tseg)[0])
    labels_val_full = np.array(gt.conversion(datadir+"grs1915_%is_labels_val.txt"%tseg)[0])


    tstart_train_full = np.loadtxt(datadir+"grs1915_%is_tstart_train.txt"%tseg)
    tstart_test_full = np.loadtxt(datadir+"grs1915_%is_tstart_test.txt"%tseg)
    tstart_val_full = np.loadtxt(datadir+"grs1915_%is_tstart_val.txt"%tseg)

    features_all_full = np.concatenate((features_train_full, features_val_full,
                                        features_test_full))


    lc_all_full = gt.getpickle(datadir+"grs1915_%is_lc_all.dat"%tseg)
    hr_all_full = gt.getpickle(datadir+"grs1915_%is_hr_all.dat"%tseg)

    lc_train_full = lc_all_full["train"]
    lc_test_full = lc_all_full["test"]
    lc_val_full = lc_all_full["val"]

    hr_train_full = hr_all_full["train"]
    hr_test_full = hr_all_full["test"]
    hr_val_full = hr_all_full["val"]

    # NOTE: This line only works unless I change the features or the
    # order of the features! BEWARE!
    # It takes care of a single segment that seems to be an outlier in
    # terms of the hardness ratios
    # It's strange enough to be fairly certain it's instrumental
    delete_ind = np.where(features_all_full[:,17] >= 20)[0]

    print("delete_ind: " + str(delete_ind))

    if delete_ind > len(features_train_full):
        delete_ind_new = delete_ind - len(features_train_full)
    else:
        delete_ind_new = delete_ind
        delete_set = "train"

    if delete_ind_new > len(features_val_full):
        delete_ind_new -= len(features_val_full)
        delete_set = "test"
    else:
        delete_set = "val"

    print("delete_ind_new: " + str(delete_ind_new))
    print("delete_set: " + str(delete_set))

    if delete_set == "train":
        features_train_full = np.delete(features_train_full,
                                        delete_ind_new, axis=0)
        labels_train_full = np.delete(labels_train_full,
                                        delete_ind_new, axis=0)
        lc_train_full = np.delete(lc_train_full,
                                        delete_ind_new, axis=0)
        hr_train_full = np.delete(hr_train_full,
                                        delete_ind_new, axis=0)

    elif delete_set == "val":
        features_val_full = np.delete(features_val_full,
                                        delete_ind_new, axis=0)
        labels_val_full = np.delete(labels_val_full,
                                        delete_ind_new, axis=0)
        lc_val_full = np.delete(lc_val_full,
                                        delete_ind_new, axis=0)
        hr_val_full = np.delete(hr_val_full,
                                        delete_ind_new, axis=0)

    elif delete_set == "test":
        features_test_full = np.delete(features_test_full,
                                        delete_ind_new, axis=0)
        labels_test_full = np.delete(labels_test_full,
                                        delete_ind_new, axis=0)
        lc_test_full = np.delete(lc_test_full,
                                        delete_ind_new, axis=0)
        hr_test_full = np.delete(hr_test_full,
                                        delete_ind_new, axis=0)

    features_all_full = np.concatenate((features_train_full, features_val_full,
                                        features_test_full))


    print("after removal: " + str(np.where(features_all_full[:,17] >= 20)[0]))

    if log_features is None:
        log_features = [2, 5, 6, 7, 9, 10, 11, 14, 16]

    for l in log_features:
        features_train_full[:,l] = np.log(features_train_full[:,l])
        features_test_full[:,l] = np.log(features_test_full[:,l])
        features_val_full[:,l] = np.log(features_val_full[:,l])


    labels_train_full = transform_chi(labels_train_full)
    labels_test_full = transform_chi(labels_test_full)
    labels_val_full = transform_chi(labels_val_full)

    features = {"train": features_train_full,
                "test": features_test_full,
                "val": features_val_full}

    labels = {"train": labels_train_full,
              "test": labels_test_full,
              "val": labels_val_full}

    lc = {"train": lc_train_full,
          "test": lc_test_full,
          "val": lc_val_full}

    hr = {"train": hr_train_full,
          "test": hr_test_full,
          "val": hr_val_full}

    return features, labels, lc, hr


def labelled_data(features, labels, lc, hr):

    labels_train_full = labels["train"]
    labels_test_full = labels["test"]
    labels_val_full = labels["val"]

    train_ind =  np.where(labels_train_full != "None")[0]
    test_ind =  np.where(labels_test_full != "None")[0]
    val_ind =  np.where(labels_val_full != "None")[0]

    labels_train = labels_train_full[train_ind]
    labels_test = labels_test_full[test_ind]
    labels_val = labels_val_full[val_ind]

    features_train = features["train"][train_ind]
    features_test = features["test"][test_ind]
    features_val = features["val"][val_ind]

    lc_train = np.array([lc["train"] for i in train_ind])
    lc_test = np.array([lc["test"] for i in test_ind])
    lc_val = np.array([lc["val"] for i in val_ind])

    hr_train = np.array([hr["train"] for i in train_ind])
    hr_test = np.array([hr["test"] for i in test_ind])
    hr_val = np.array([hr["val"] for i in val_ind])

    features_lb = {"train": features_train,
                "test": features_test,
                "val": features_val}

    labels_lb = {"train": labels_train,
              "test": labels_test,
              "val": labels_val}

    lc_lb = {"train": lc_train,
          "test": lc_test,
          "val": lc_val}

    hr_lb = {"train": hr_train,
          "test": hr_test,
          "val": hr_val}

    return features_lb, labels_lb, lc_lb, hr_lb


def scale_features(features, features_lb=None):

    features_all = np.hstack([features["train"],
                              features["val"],
                              features["test"]])

    scaler = StandardScaler().fit(features_all)

    fscaled = {"train": scaler.transform(features["train"]),
               "test": scaler.transform(features["test"]),
               "val": scaler.transform(features["val"])}

    if features_lb is not None:
        fscaled_lb = {"train": scaler.transform(features_lb["train"]),
                      "test": scaler.transform(features_lb["test"]),
                      "val": scaler.transform(features_lb["val"])}

        return fscaled, fscaled_lb

    else:
        return fscaled


def greedy_search(datadir, seg_length_supervised=1024.):


    features, labels, lc, hr = load_features(datadir, seg_length_supervised)

    features_lb, labels_lb, lc_lb, hr_lb = labelled_data(features, labels,
                                                         lc, hr)

    features_train = features_lb["train"]
    features_val = features_lb["val"]
    features_test = features_lb["test"]
    labels_train = labels_lb["train"]
    labels_val =  labels_lb["val"]
    labels_test = labels_lb["test"]

    score_all = [] 
    feature_ranking = []
    nfeatures = range(features_train.shape[1])
    features_new_train = []
    features_new_val = []
    features_new_test = []
    best_params_all = []

    for i in range(features_train.shape[1]):
        print("I am on the %ith loop"%i)
        score = []
        best_params = []
        ## first feature
        for j in nfeatures:
            if j in feature_ranking:
                continue
            #print("I am on feature %i"%j)
            if len(features_new_train) == 0:
                ft = np.atleast_2d(features_train[:,j]).T
                fv = np.atleast_2d(features_val[:,j]).T
                fte = np.atleast_2d(features_test[:,j]).T
            else:
                ft = np.concatenate((features_new_train, ft), 1)
                fv = np.concatenate((features_new_val, fv), 1)
                fte = np.concatenate((features_new_test, fte), 1)
            ### scale features
            f_all = np.concatenate((ft, fv, fte))

            scaler_train = StandardScaler().fit(f_all)
            fscaled_train = scaler_train.transform(ft)
        
            fscaled_val = scaler_train.transform(fv)
            ### Random Forest Classifier
            params = {'max_depth': [10,50,100,200,400]}#,
            grid_rfc = GridSearchCV(RandomForestClassifier(n_estimators=250),
                                    param_grid=params, verbose=0, n_jobs=15)

            grid_rfc.fit(fscaled_train, labels_train)
            best_params.append(grid_rfc.best_params_)
            score.append(grid_rfc.score(fscaled_val, labels_val))
    
        score_all.append(score)
        best_params_all.append(best_params)
        best_ind = np.where(score == np.max(score))[0]
        print("best_ind: " + str(best_ind))
        if len(best_ind) > 1:
            best_ind = best_ind[0]
        print("The best score in round " + str(i) + " is " + str(np.max(score)))
        n_best = nfeatures.pop(best_ind)
        print("The best-ranked feature in round " + str(i) + " is " + str(n_best))
        feature_ranking.append(n_best)
        if len(features_new_train) == 0:
            features_new_train = np.atleast_2d(features_train[:,n_best]).T
            features_new_val = np.atleast_2d(features_val[:,n_best]).T
            features_new_test = np.atleast_2d(features_test[:,n_best]).T
        else:
            features_new_train = np.concatenate((features_new_train,
                                                 np.atleast_2d(features_train[:,n_best]).T), 1)
            features_new_val = np.concatenate((features_new_val,
                                               np.atleast_2d(features_val[:,n_best]).T), 1)
            features_new_test = np.concatenate((features_new_test,
                                                np.atleast_2d(features_test[:,n_best]).T), 1)

    features_new_train = np.array([features["train"][:,i] for i in feature_ranking]).T
    features_new_test = np.array([features["test"][:,i] for i in feature_ranking]).T
    features_new_val = np.array([features["val"][:,i] for i in feature_ranking]).T


    res = {"ranking":feature_ranking, "fnew_train":features_new_train,
           "fnew_val":features_new_val, "fnew_test":features_new_test,
           "scores":score_all}  

    f = open(datadir+"grs1915_greedysearch_res.dat", "w")
    pickle.dump(res, f, -1)
    f.close()

    return

def main():
    datadir= "/scratch/daniela/data/grs1915/"
    seg_length_supervised = 1024.

    greedy_search(datadir, seg_length_supervised=seg_length_supervised)

    return

if __name__ == "__main__":
    main()
