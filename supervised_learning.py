__author__ = 'danielahuppenkothen'




import matplotlib.pyplot as plt
import feature_extraction

import glob
import cPickle as pickle
import numpy as np






from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV


from sklearn.ensemble import RandomForestClassifier

from sklearn import linear_model
from sklearn.svm import LinearSVC



def supervised_all(datadir="../"):

    files = glob.glob(datadir+"*features.dat")

    for f in files:
        fin =  open(f, 'r')
        features_all = pickle.load(fin)
        fin.close()

        ### fun with file names
        froot = f.split("_")
        seglength = np.float(froot[1])
        ftype = froot[2]

        if ftype == "summary":
            print("Running on summary of features with segment length of %i seconds"%int(seglength))
        elif ftype == "hrfull":
            print("Running on HR maps and summary features with segment length of %i seconds"%int(seglength))
        elif ftype == "psfull":
            print("Running on full periodogram and summary of features with segment length of %i seconds"%int(seglength))
        else:
            print("Something's gone very wrong!")


    run_supervised(features_all)

    return

def run_supervised(features_all):
        """
        features_all is a dictionary that contains keywords "test", "train" and "val".
        Each keyword contains a list. The first list element is another dictionary with keywords
        "features" (containing the feature vectors), "lc" (containing the light curve) and "hr" 
        (containing the hardness ratios). The second list item is a list of labels for each segment.
        """

        features_train = np.array(features_all["train"][0]["features"])
        features_val = np.array(features_all["val"][0][0]["features"])
        features_test = np.array(features_all["test"][0]["features"])
        print("features_train.shape: " + str(features_train.shape))
        labels_train = features_all["train"][1]
        labels_val = features_all["val"][0][1]
        labels_test = features_all["test"][1]


        lc_train = features_all["train"][0]["lc"]
        lc_val = features_all["val"][0][0]["lc"]
        lc_test = features_all["test"][0]["lc"]

        hr_train = features_all["train"][0]["hr"]
        hr_val = features_all["val"][0][0]["hr"]
        hr_test = features_all["test"][0]["hr"]


        ### scale features
        scaler_train = StandardScaler().fit(features_train)
        fscaled_train = scaler_train.transform(features_train)

        scaler_val = StandardScaler().fit(features_val)
        fscaled_val = scaler_val.transform(features_val)


        ### simplest algorithm: K-Nearest Neighbour
        params = {'n_neighbors': [1, 3, 5, 10, 15, 20, 25, 30, 50]}#, 'max_features': }
        grid = GridSearchCV(KNeighborsClassifier(), param_grid=params, verbose=10, n_jobs=10)
        grid.fit(fscaled_train, labels_train)

        print("Best results for the K Nearest Neighbour run:")
        print("Best parameter: " + str(grid.best_params_))
        print("Training accuracy: " + str(grid.score(fscaled_train, labels_train)))
        print("Validation accuracy: " + str(grid.score(fscaled_val, labels_val)))


        ### Random Forest Classifier
        params = {'max_depth': [7, 10,12, 15,17, 18, 19, 20, 21, 22, 23, 25,30,40, 50, 100, 200, 500]}#,
                 # 'max_features':[2,3,4,5,6,7,8,10,50,150,200,250,300]}
        grid_rfc = GridSearchCV(RandomForestClassifier(n_estimators=500), param_grid=params,
                                verbose=10, n_jobs=10)
        grid_rfc.fit(fscaled_train, labels_train)

        print("Best results for the Random Forest run:")
        print("Best parameter: " + str(grid_rfc.best_params_))
        print("Training accuracy: " + str(grid_rfc.score(fscaled_train, labels_train)))
        print("Validation accuracy: " + str(grid_rfc.score(fscaled_val, labels_val)))


        ### Linear Classifier
        params = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
        grid_lm = GridSearchCV(linear_model.LogisticRegression(penalty="l2", class_weight="auto"),
                            param_grid=params, verbose=10, n_jobs=10)
        grid_lm.fit(fscaled_train, labels_train)
        #params = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
        #grid = GridSearchCV(LinearSVC(), param_grid=params, verbose=10)
        #grid.fit(features_train, labels_train)


        print("Best results for the Linear Model run:")
        print("Best parameter: " + str(grid_lm.best_params_))
        print("Training accuracy: " + str(grid_lm.score(fscaled_train, labels_train)))
        print("Validation accuracy: " + str(grid_lm.score(fscaled_val, labels_val)))

        return grid, grid_rfc, grid_lm
