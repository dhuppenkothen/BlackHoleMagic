__author__ = 'danielahuppenkothen'


import numpy as np
import generaltools as gt
## scale features to standard mean and variance
from sklearn.preprocessing import StandardScaler



def load_data(datadir="./", hr=False):
    """
    Helper function for the GRS1915_TSNEViz notebook that does the
    mpld3 visualization of the data.

    :param datadir: directory with the data
    :return: you know, stuff
    """

## Load features and label: grs1915_1024s_features_train.txts
    features_train = np.loadtxt(datadir+"grs1915_1024s_features_train.txt")
    features_val = np.loadtxt(datadir+"grs1915_1024s_features_val.txt")
    features_test = np.loadtxt(datadir+"grs1915_1024s_features_test.txt")

    labels_train = gt.conversion(datadir+"grs1915_1024s_labels_train.txt")[0]
    labels_val = gt.conversion(datadir+"grs1915_1024s_labels_val.txt")[0]
    labels_test = gt.conversion(datadir+"grs1915_1024s_labels_test.txt")[0]


    ## extract features according to ranking
    feature_ranking = [13, 8, 2, 3, 1, 9, 11, 4, 12, 0, 15, 14, 7, 10, 6, 5]
    ## number of new features
    nfeatures = 12

    ## shorten feature_ranking
    feature_ranking = feature_ranking[:nfeatures]

    ## make empty arrays for the new feature vectors
    features_new_train = np.zeros((features_train.shape[0],len(feature_ranking)))
    features_new_val = np.zeros((features_val.shape[0],len(feature_ranking)))
    features_new_test = np.zeros((features_test.shape[0],len(feature_ranking)))

    ## fill empty arrays with the right features
    for i,f in enumerate(feature_ranking):
        features_new_train[:,i] = features_train[:,f]
        features_new_val[:,i] = features_val[:,f]
        features_new_test[:,i] = features_test[:,f]

    print(features_new_train.shape)
    print(features_new_val.shape)
    print(features_new_test.shape)

    features_new_all = np.concatenate((features_new_train, features_new_val, features_new_test))
    print(features_new_all.shape)
    labels_all = np.concatenate((labels_train, labels_val, labels_test))


    scaler_train = StandardScaler().fit(features_new_all)
    fscaled_train = scaler_train.transform(features_new_train)
    fscaled_val = scaler_train.transform(features_new_val)
    fscaled_test = scaler_train.transform(features_new_test)
    fscaled_all = scaler_train.transform(features_new_all)


    ## load t-SNE results
    tsne_loaded = gt.getpickle(datadir+"grs1915_tsne_all.dat")
    tsne, asdf, labels_all = tsne_loaded
    label_set = np.array(list(set(labels_all)))

    ## make subsample with only labelled data
    label_ind = np.where(labels_all != "None")
    features_red = features_new_all[label_ind]
    labels_red = labels_all[label_ind]
    asdf_red = asdf[label_ind]

    lc_all = gt.getpickle("/Volumes/Lliarinh/data/grs1915/grs1915_1024_all_summary_lc_all.dat")
    lc_test = np.array(lc_all["test"])
    lc_val = np.array(lc_all["val"])
    lc_train = np.array(lc_all["train"])
    lc_test_new = np.array([l.T for l in lc_test])
    lc_train_new = np.array([l.T for l in lc_train])
    lc_val_new = np.array([l.T for l in lc_val])
    print(lc_test_new.shape)
    print(lc_train_new.shape)
    print(lc_val_new.shape)

    lc_all_new = np.concatenate((lc_train_new, lc_val_new, lc_test_new))

    lc_all_rebinned = []

    for l in lc_all_new:
        #print(l.shape)
        #print(l[0])
        bintimes, bincounts = gt.rebin_lightcurve(l[:,0]-l[0,0], l[:,1], n=10, type='average')
        lc_all_rebinned.append(np.array([bintimes, bincounts]).T)

    lc_all_rebinned = np.array(lc_all_rebinned)
    lc_all_red = lc_all_rebinned[label_ind]

    if hr:
         hr_all = gt.getpickle("/Volumes/Lliarinh/data/grs1915/grs1915_1024_all_summary_hr_all.dat")

         hr_test = np.array(hr_all["test"])
         hr_val = np.array(hr_all["val"])
         hr_train = np.array(hr_all["train"])
         print(lc_test_new.shape)
         print(lc_train_new.shape)
         print(lc_val_new.shape)

         hr_all_new = np.concatenate((hr_train, hr_val, hr_test))
         hr_all_red = hr_all_new[label_ind]

         return asdf, asdf_red, labels_red, lc_all_rebinned, lc_all_red, hr_all_red

    else:
        return asdf, asdf_red, labels_red, lc_all_rebinned, lc_all_red
