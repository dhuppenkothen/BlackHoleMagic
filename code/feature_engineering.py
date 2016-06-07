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


def greedy_search(datadir, seg_length_supervised=1024.):


    features_train_full = np.loadtxt(datadir+"grs1915_%is_features_train.txt"%seg_length_supervised)
    features_test_full = np.loadtxt(datadir+"grs1915_%is_features_test.txt"%seg_length_supervised)
    features_val_full = np.loadtxt(datadir+"grs1915_%is_features_val.txt"%seg_length_supervised)

    labels_test_full = np.array(gt.conversion(datadir+"grs1915_%is_labels_test.txt"%seg_length_supervised)[0])
    labels_train_full = np.array(gt.conversion(datadir+"grs1915_%is_labels_train.txt"%seg_length_supervised)[0])
    labels_val_full = np.array(gt.conversion(datadir+"grs1915_%is_labels_val.txt"%seg_length_supervised)[0])

    labels_train_full = transform_chi(labels_train_full)
    labels_test_full = transform_chi(labels_test_full)
    labels_val_full = transform_chi(labels_val_full)

    tstart_train_full = np.loadtxt(datadir+"grs1915_%is_tstart_train.txt"%seg_length_supervised)
    tstart_test_full = np.loadtxt(datadir+"grs1915_%is_tstart_test.txt"%seg_length_supervised)
    tstart_val_full = np.loadtxt(datadir+"grs1915_%is_tstart_val.txt"%seg_length_supervised)

    features_all_full = np.concatenate((features_train_full, features_val_full, features_test_full))

    features_train = features_train_full[np.where(labels_train_full != "None")]
    features_test = features_test_full[np.where(labels_test_full != "None")]
    features_val= features_val_full[np.where(labels_val_full != "None")]

    labels_train= labels_train_full[np.where(labels_train_full != "None")]
    labels_test = labels_test_full[np.where(labels_test_full != "None")]
    labels_val = labels_val_full[np.where(labels_val_full != "None")]

    labels_train = transform_chi(labels_train)
    labels_test = transform_chi(labels_test)
    labels_val = transform_chi(labels_val)
 
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
            #print("NaN in row: " + str(np.where(np.isnan(f_all))))
            scaler_train = StandardScaler().fit(f_all)
            fscaled_train = scaler_train.transform(ft)
        
            fscaled_val = scaler_train.transform(fv)
            ### Random Forest Classifier
            params = {'max_depth': [10,50,100,200,400]}#,
            grid_rfc = GridSearchCV(RandomForestClassifier(n_estimators=250), param_grid=params,
                                verbose=0, n_jobs=15)
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
            features_new_train = np.concatenate((features_new_train, np.atleast_2d(features_train[:,n_best]).T), 1)
            features_new_val = np.concatenate((features_new_val, np.atleast_2d(features_val[:,n_best]).T), 1)
            features_new_test = np.concatenate((features_new_test, np.atleast_2d(features_test[:,n_best]).T), 1)


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

    greedy_search(datadir, seg_length_supervised=1024.)

    return

if __name__ == "__main__":
    main()
