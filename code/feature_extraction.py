
import matplotlib.pyplot as plt
import numpy as np
import glob
import copy
import cPickle as pickle
import pandas as pd
import astroML.stats
import scipy.stats
from sklearn.preprocessing import StandardScaler


import linearfilter

## set the seed for the random number generator
## such that my "random choices" remain reproducible
np.random.seed(20150608)


import lightcurve
import powerspectrum

import powerspectrum


def split_dataset(d_all, train_frac = 0.5, validation_frac = 0.25, test_frac = 0.25):

    ## total number of light curves
    n_lcs = len(d_all)

    ## shuffle list of light curves
    np.random.shuffle(d_all)

    #train_frac = 0.5
    #validation_frac = 0.25
    #test_frac = 0.25

    ## let's pull out light curves for three data sets into different variables.
    d_all_train = d_all[:int(train_frac*n_lcs)]
    d_all_val = d_all[int(train_frac*n_lcs):int((train_frac + validation_frac)*n_lcs)]
    d_all_test = d_all[int((train_frac + validation_frac)*n_lcs):]

    ## Let's print some information about the three sets.
    print("There are %i light curves in the training set."%len(d_all_train))
    print("There are %i light curves in the validation set."%len(d_all_val))
    print("There are %i light curves in the test set."%len(d_all_test))
    for da,n in zip([d_all_train, d_all_val, d_all_test], ["training", "validation", "test"]):
        print("These is the distribution of states in the %s set: "%n)
        states = [d[1] for d in da]
        st = pd.Series(states)
        print(st.value_counts())
        print("================================================================")

    return d_all_train, d_all_val, d_all_test


## This function is also in grs1915_utils.py!
def extract_segments(d_all, seg_length = 256., overlap=64.):
    """ Extract light curve segments from a list of light curves.
        Each element in the list is a list with two elements:
        - an array that contains the light curve in three energy bands
        (full, low energies, high energies) and
        - a string containing the state of that light curve.

        The parameters are
        - seg_length: the length of each segment. Bits of data at the end of a light curve
        that are smaller than seg_length will not be included.
        - overlap: This is actually the interval between start times of individual segments,
        i.e. by default light curves start 64 seconds apart. The actual overlap is
        seg_length-overlap
    """
    segments, labels, nsegments = [], [], [] ## labels for labelled data

    for i,d_seg in enumerate(d_all):

        ## data is an array in the first element of d_seg
        data = d_seg[0]
        ## state is a string in the second element of d_seg
        state = d_seg[1]

        ## if the length of the light curve is shorter than the
        ## chosen segment length, discard this light curve
        length = data[-1,0] - data[0,0]
        if length < seg_length:
            continue

        ## compute the intervals between adjacent time bins
        dt_data = data[1:,0] - data[:-1,0]

        dt = np.min(dt_data)

        ## compute the number of time bins in a segment
        nseg = seg_length/dt
        ## compute the number of time bins to start of next segment
        noverlap = overlap/dt

        istart = 0
        iend = nseg
        j = 0

        while iend <= len(data):
            dtemp = data[istart:iend]
            segments.append(dtemp)
            labels.append(state)
            istart += noverlap
            iend += noverlap
            j+=1

        nsegments.append(j)

    ## convert labels to strings so I don't have a weird
    ## combination of None objects and strings
    labels = np.array([str(l) for l in labels])

    return segments, labels, np.array(nsegments)

def extract_data(d_all, val=True, train_frac=0.5, validation_frac=0.25, test_frac = 0.25,
                  seg=True, seg_length=1024., overlap = 128.):


    #f = open(filename)
    #d_all = pickle.load(f)
    #f.close()

    ## print light curve statistics
    print("Number of light curves:  " + str(len(d_all)))
    states = [d[1] for d in d_all]
    st = pd.Series(states)
    st.value_counts()

    d_all_train, d_all_val, d_all_test = split_dataset(d_all)

    if not val:
        d_all_train  = d_all_train + d_all_val

    if seg:
        seg_train, labels_train, nseg_train = extract_segments(d_all_train, seg_length=seg_length, overlap=overlap)
        seg_test, labels_test, nseg_test = extract_segments(d_all_test, seg_length=seg_length, overlap=overlap)

        if val:
            seg_val, labels_val, nseg_val = extract_segments(d_all_val, seg_length=seg_length, overlap=overlap)


    else:
        seg_train = [d[0] for d in d_all_train]
        labels_train = [d[1] for d in d_all_train]
        nseg_train = [1 for d in d_all_train]

        seg_test = [d[0] for d in d_all_test]
        labels_test = [d[1] for d in d_all_test]
        nseg_test = [1 for d in d_all_test]

        if val:
            seg_val = [d[0] for d in d_all_val]
            labels_val = [d[1] for d in d_all_val]
            nseg_val = [1 for d in d_all_val]


        ## Let's print some details on the different segment data sets
        print("There are %i segments in the training set."%len(seg_train))
        if val:
            print("There are %i segments in the validation set."%len(seg_val))
        print("There are %i segments in the test set."%len(seg_test))
        if val:
            labelset = [labels_train, labels_val, labels_test]
            keys = ["training", "validation", "test"]
        else:
            labelset = [labels_train, labels_test]
            keys = ["training", "test"]

        for la,n in zip(labelset, keys):
            print("These is the distribution of states in the %s set: "%n)
            st = pd.Series(la)
            print(st.value_counts())
            print("================================================================")

    if val:
        return [[seg_train, labels_train, nseg_train],
                [seg_val, labels_val, nseg_val],
                [seg_test, labels_test, nseg_test]]
    else:
        return [[seg_train, labels_train, nseg_train],
                [seg_test, labels_test, nseg_test]]

######################################################################################################################
#### FUNCTIONS FOR FEATURE EXTRACTION ################################################################################
######################################################################################################################

### FIRST PART: TIME SERIES FEATURES

def timeseries_features(seg):
    counts = seg[:,1]
    fmean = np.mean(counts)
    fmedian = np.median(counts)
    fvar = np.var(counts)
    skew = scipy.stats.skew(counts)
    kurt = scipy.stats.kurtosis(counts)
    return fmean, fmedian, fvar, skew, kurt

def linear_filter(counts_scaled, k=10, lamb=None):
    """
    Extract features from a linear filter.
    :param counts_scaled: numpy ndarray (nsamples, ntimebins) with SCALED TOTAL COUNTS
    :return ww_all: numpy ndarray (nsamples, k) of weights
    """

    ## initialize the LinearFilter object
    lf = linearfilter.LinearFilter(k=k, lamb=lamb)

    ww_all = np.zeros((len(counts_scaled), k))

    ## loop through all light curves and compute the weight vector for each
    for i,c in enumerate(counts_scaled):
        ww_all[i,:] = lf.fit_transform(c)[0]
    return ww_all




#### FOURIER DOMAIN FEATURES


## boundaries for power bands
pcb = {"pa_min":0.0039, "pa_max":0.031,
       "pb_min":0.031, "pb_max":0.25,
       "pc_min":0.25, "pc_max":2.0,
       "pd_min":2.0, "pd_max":16.0}

def rebin_psd(freq, ps, n=10, type='average'):

    nbins = int(len(freq)/n)
    df = freq[1] - freq[0]
    T = freq[-1] - freq[0] + df
    bin_df = df*n
    binfreq = np.arange(nbins)*bin_df + bin_df/2.0 + freq[0]

    #print("len(ps): " + str(len(ps)))
    #print("n: " + str(n))

    nbins_new = int(len(ps)/n)
    ps_new = ps[:nbins_new*n]
    binps = np.reshape(np.array(ps_new), (nbins_new, int(n)))
    binps = np.sum(binps, axis=1)
    if type in ["average", "mean"]:
        binps = binps/np.float(n)
    else:
        binps = binps

    if len(binfreq) < len(binps):
        binps= binps[:len(binfreq)]

    return binfreq, binps




def psd_features(seg, pcb):
    """
    Computer PSD-based features.
    seg: data slice of type [times, count rates, count rate error]^T
    pcb: frequency bands to use for power colours
    """

    times = seg[:,0]
    dt = times[1:] - times[:-1]
    dt = np.min(dt)

    counts = seg[:,1]*dt
    #ps = powerspectrum.PowerSpectrum(times, counts=counts, norm="rms")
    freq, ps = make_psd(seg, navg=1)
    #print("len(ps), before: " + str(len(ps)))
    if times[-1]-times[0] >= 2.*128.0:
        tlen = (times[-1]-times[0])
        nrebin = np.round(tlen/128.)
        freq, ps = rebin_psd(freq, ps, n=nrebin, type='average')

    #print("len(ps), after: " + str(len(ps)))

    freq = np.array(freq[1:])
    ps = ps[1:]
    #print("min(freq): " + str(np.min(freq)))
    #print("max(freq): " + str(np.max(freq)))
    #print("len(freq): " + str(len(freq)))

    binfreq, binps = total_psd(seg, 24)

    #print("min(binps): " + str(np.min(binps)))
    #print("max(binps): " + str(np.max(binps)))
    fmax_ind = np.where(binps == np.max(binps))
    #print("fmax_ind: " + str(fmax_ind))
    maxfreq = binfreq[fmax_ind[0]]
    #print("maxfreq: " + str(maxfreq))

    ## find power in spectral bands for power-colours
    pa_min_freq = freq.searchsorted(pcb["pa_min"])
    pa_max_freq = freq.searchsorted(pcb["pa_max"])

    pb_min_freq = freq.searchsorted(pcb["pb_min"])
    pb_max_freq = freq.searchsorted(pcb["pb_max"])

    pc_min_freq = freq.searchsorted(pcb["pc_min"])
    pc_max_freq = freq.searchsorted(pcb["pc_max"])

    pd_min_freq = freq.searchsorted(pcb["pd_min"])
    pd_max_freq = freq.searchsorted(pcb["pd_max"])

    psd_a = np.sum(ps[pa_min_freq:pa_max_freq])
    psd_b = np.sum(ps[pb_min_freq:pb_max_freq])
    psd_c = np.sum(ps[pc_min_freq:pc_max_freq])
    psd_d = np.sum(ps[pd_min_freq:pd_max_freq])
    pc1 = np.sum(ps[pc_min_freq:pc_max_freq])/np.sum(ps[pa_min_freq:pa_max_freq])
    pc2 = np.sum(ps[pb_min_freq:pb_max_freq])/np.sum(ps[pd_min_freq:pd_max_freq])

    return maxfreq, psd_a, psd_b, psd_c, psd_d, pc1, pc2

def make_psd(segment, navg=1):

    times = segment[:,0]
    dt = times[1:] - times[:-1]
    dt = np.min(dt)

    counts = segment[:,1]*dt

    tseg = times[-1]-times[0]
    nlc = len(times)
    nseg = int(nlc/navg)

    if navg == 1:
        ps = powerspectrum.PowerSpectrum(times, counts=counts, norm="rms")
        ps.freq = np.array(ps.freq)
        ps.ps = np.array(ps.ps)*ps.freq
        return ps.freq, ps.ps
    else:
        ps_all = []
        for n in xrange(navg):
            t_small = times[n*nseg:(n+1)*nseg]
            c_small = counts[n*nseg:(n+1)*nseg]
            ps = powerspectrum.PowerSpectrum(t_small, counts=c_small, norm="rms")
            ps.freq = np.array(ps.freq)
            ps.ps = np.array(ps.ps)*ps.freq
            ps_all.append(ps.ps)

        #print(np.array(ps_all).shape)

        ps_all = np.average(np.array(ps_all), axis=0)

        #print(ps_all.shape)

    return ps.freq, ps_all

epsilon = 1.e-8

def total_psd(seg, bins):
    times = seg[:,0]
    dt = times[1:] - times[:-1]
    dt = np.min(dt)
    counts = seg[:,1]*dt

    ps = powerspectrum.PowerSpectrum(times, counts=counts, norm="rms")
    ps.ps = np.array(ps.freq)*np.array(ps.ps)
    binfreq = np.logspace(np.log10(ps.freq[1]-epsilon), np.log10(ps.freq[-1]+epsilon), bins)
    #print("freq: " + str(ps.freq[1:10]))
    #print("binfreq: " + str(binfreq[:10]))
    binps, bin_edges, binno = scipy.stats.binned_statistic(ps.freq[1:], ps.ps[1:], statistic="mean", bins=binfreq)
    df = binfreq[1:]-binfreq[:-1]
    binfreq = binfreq[:-1]+df/2.

    return np.array(binfreq), np.array(binps)



def compute_hrlimits(hr1, hr2):
    min_hr1 = np.min(hr1)
    max_hr1 = np.max(hr1)

    min_hr2 = np.min(hr2)
    max_hr2 = np.max(hr2)
    return [[min_hr1, max_hr1], [min_hr2, max_hr2]]

def hr_maps(seg, bins=30, hrlimits=None):
    times = seg[:,0]
    counts = seg[:,1]
    low_counts = seg[:,2]
    mid_counts = seg[:,3]
    high_counts = seg[:,4]
    hr1 = np.log(mid_counts/low_counts)
    hr2 = np.log(high_counts/low_counts)

    if hrlimits is None:
        hr_limits = compute_hrlimits(hr1, hr2)
    else:
        hr_limits = hrlimits

    h, xedges, yedges = np.histogram2d(hr1, hr2, bins=bins,
                                       range=hr_limits)
    h = np.rot90(h)
    h = np.flipud(h)
    hmax = np.max(h)
    #print(hmax)
    hmask = np.where(h > hmax/20.)
    hmask1 = np.where(h < hmax/20.)
    hnew = copy.copy(h)
    hnew[hmask[0], hmask[1]] = 1.
    hnew[hmask1[0], hmask1[1]] = 0.0
    return xedges, yedges, hnew

def hr_fitting(seg):
    counts = seg[:,1]
    low_counts = seg[:,2]
    mid_counts = seg[:,3]
    high_counts = seg[:,4]
    hr1 = mid_counts/low_counts
    hr2 = high_counts/low_counts

    # compute the robust statistics
    #(mu_r, sigma1_r,
    # sigma2_r, alpha_r) = astroML.stats.fit_bivariate_normal(hr1, hr2, robust=True)
    #if any(np.isnan(mu_r)) or np.isnan(sigma1_r) or np.isnan(sigma2_r) or np.isnan(alpha_r):
    #    print("mu_r: " + str(mu_r))
    #    print("sigma1_r: " + str(sigma1_r))
    #    print("sigma2_r: " + str(sigma2_r))
    #    print("alpha_r: " + str(alpha_r))
    hr1_mask = np.where(np.isinf(hr1) == False)
    #print(np.where(np.isinf(hr2) == False))
    hr1 = hr1[hr1_mask]
    hr2 = hr2[hr1_mask]

    hr2_mask = np.where(np.isinf(hr2) == False)
    hr1 = hr1[hr2_mask]
    hr2 = hr2[hr2_mask]

    mu1 = np.mean(hr1)
    mu2 = np.mean(hr2)

    cov = np.cov(hr1, hr2).flatten()

    skew = scipy.stats.skew(np.array([hr1, hr2]).T)
    kurt = scipy.stats.kurtosis(np.array([hr1, hr2]).T)

    if np.any(np.isnan(cov)):
        print("NaN in cov")

    return mu1, mu2, cov, skew, kurt
#    return mu_r, sigma1_r, sigma2_r, alpha_r

def hid_maps(seg, bins=30):
    counts = seg[:,1]
    low_counts = seg[:,2]
    high_counts = seg[:,3]
    hr2 = high_counts/low_counts
    hid_limits = compute_hrlimits(hr2, counts)

    h, xedges, yedges = np.histogram2d(hr2, counts, bins=bins,
                                       range=hid_limits)
    h = np.rot90(h)
    h = np.flipud(h)

    return xedges, yedges, h


def lcshape_features(seg, dt=1.0):

    times = seg[:,0]
    counts = seg[:,1]

    dt_small = times[1:]-times[:-1]
    dt_small = np.min(dt_small)

    nbins = np.round(dt/dt_small)

    bintimes, bincounts = rebin_psd(times, counts, nbins)

    return bincounts

def extract_lc(seg):
    times = seg[:,0]
    counts = seg[:,1]
    low_counts = seg[:,2]
    high_counts = seg[:,3]
    hr1 = low_counts/counts
    hr2 = high_counts/counts
    return [times, counts, hr1, hr2]



def make_features(seg, k=10, bins=30, lamb=None,
                  hr_summary=True, ps_summary=True, lc=True, hr=True, hrlimits=None):
    """
    Make features from a set of light curve segments, except for the linear filter!

    :param seg: list of all segments to be used
    :param bins: bins used in a 2D histogram if hr_summary is False
    :param hr_summary: if True, summarize HRs in means and covariance matrix
    :param ps_summary: if True, summarize power spectrum in frequency of maximum power and power spectral bands
    :param lc: if True, store light curves
    :param hr: if True, store hardness ratios
    :param hrlimits: limits for the 2D histogram if hr_summary is False
    :return: fdict: dictionary with keywords "features", "lc" and "hr"
    """
    features = []
    if lc:
        lc_all = []
    if hr:
        hr_all = []


    ## MAKE FEATURES BASED ON LINEAR FILTER

    ## first, extract the total counts out of each segment
    counts = np.array([s[:,1] for s in seg])

    ## next, transform the array such that *each light curve* is scaled
    ## to zero mean and unit variance
    ## We can do this for all light curves independently, because we're
    ## averaging *per light curve* and not *per time bin*
    counts_scaled = StandardScaler().fit_transform(counts.T).T

    ## transform the counts into a weight vector
    ww = linear_filter(counts_scaled, k=k)
    
    for s in seg:

        features_temp = []

        ## time series summary features
        fmean, fmedian, fvar, fskew, fkurt = timeseries_features(s)
        features_temp.extend([fmean, fmedian, fvar, fskew, fkurt])


        if ps_summary:
            ## PSD summary features
            maxfreq, psd_a, psd_b, psd_c, psd_d, pc1, pc2 = psd_features(s, pcb)
            #print("maxfreq: " + str(maxfreq))
            if len(maxfreq) == 0: 
                features_temp.extend([psd_a, psd_b, psd_c, psd_d, pc1, pc2])
            else: 
                features_temp.extend([maxfreq, psd_a, psd_b, psd_c, psd_d, pc1, pc2])
        else:
            ## whole PSD
            #freq, ps = make_psd(s,navg=navg)
            binfreq, binps = total_psd(s, 24)
            features_temp.extend(binps[1:])


        if hr_summary:
            mu1, mu2, cov, skew, kurt = hr_fitting(s)
            #print("len(features): " + str(len(features_temp)))
            features_temp.extend([mu1, mu2])
            #print(len(features_temp))
            features_temp.extend([cov[0], cov[1], cov[3]])
            #print(len(features_temp))
            features_temp.extend(list(skew))
            #print(len(features_temp))
            features_temp.extend(list(kurt))
            #print(len(features_temp))


        else:
            xedges, yedges, h = hr_maps(s, bins=bins, hrlimits=hrlimits)
            features_temp.extend(h.flatten())

        features.append(features_temp)


        if lc or hr:
            lc_temp = extract_lc(s)
        if lc:
            #print("appending light curve")
            lc_all.append([lc_temp[0], lc_temp[1]])
        if hr:
            #print("appending hardness ratios")
            hr_all.append([lc_temp[2], lc_temp[3]])

    print("ww.shape: " + str(np.array(ww).shape))
    print("features.shape: " + str(features[0]))
    print("type(ww): " + str(type(np.array(ww))))
    features_all = np.hstack((np.array(features), np.array(ww)))

    print("I am about to make a dictionary")
    fdict = {"features": features_all}
    print(fdict.keys)
    if lc:
        print("I am in lc")
        #features.append(lc_all)
        fdict["lc"] = lc_all
    if hr:
        print("I am in hr")
        #features.append(hr_all)
        fdict["hr"] = hr_all
    #print(fdict.keys())
    return fdict


def check_nan(features, labels, hr=True, lc=True):
    inf_ind = []
    fnew, lnew, tnew = [], [], []

    nseg = features["nseg"]
    nsum = np.array([np.sum(nseg[:i])-1 for i in xrange(1, len(nseg)+1)])
    if lc:
        lcnew = []
    if hr:
        hrnew = []

    for i,f in enumerate(features["features"]):

        try:
            if any(np.isnan(f)):
                print("NaN in sample row %i"%i)
                aind = np.array(nsum).searchsorted(i)
                nseg[aind] -= 1
                continue
            elif any(np.isinf(f)):
                print("inf sample row %i"%i)
                aind = np.array(nsum).searchsorted(i)
                nseg[aind] -= 1
                continue
            else:
                fnew.append(f)
                lnew.append(labels[i])
                tnew.append(features["tstart"][i])
                if lc:
                    lcnew.append(features["lc"][i])
                if hr:
                    hrnew.append(features["hr"][i])
        except ValueError:
            print("f: " + str(f))
            print("type(f): " + str(type(f)))
            raise Exception("This is breaking! Boo!")
    features_new = {"features":fnew, "tstart":tnew, "nseg":nseg}
    if lc:
        features_new["lc"] = lcnew
    if hr:
        features_new["hr"] = hrnew
    return features_new, lnew




def make_all_features(d_all, k=10, lamb=0.1,
                      val=True, train_frac=0.6, validation_frac=0.2, test_frac = 0.2,
                      seg=True, seg_length=1024., overlap = 64.,
                      bins=30, navg=4, hr_summary=True, ps_summary=True, lc=True, hr=True,
                      save_features=True, froot="grs1915"):

    ## Set the seed to I will always pick out the same light curves.
    np.random.seed(20150608)

    ## shuffle list of light curves
    np.random.shuffle(d_all)

    n_lcs = len(d_all)

    ## let's pull out light curves for three data sets into different variables.
    d_all_train = d_all[:int(train_frac*n_lcs)]
    d_all_test = d_all[int(train_frac*n_lcs):int((train_frac + test_frac)*n_lcs)]

    seg_train, labels_train, nseg_train = extract_segments(d_all_train, seg_length=seg_length,
                                                           overlap=overlap)
    seg_test, labels_test, nseg_test = extract_segments(d_all_test, seg_length=seg_length,
                                                        overlap=overlap)

    tstart_train = np.array([s[0,0] for s in seg_train])
    tstart_test = np.array([s[0,0] for s in seg_test])


    if val:
        d_all_val = d_all[int((train_frac + test_frac)*n_lcs):]
        seg_val, labels_val, nseg_val = extract_segments(d_all_val, seg_length=seg_length,
                                                         overlap=overlap)
        tstart_val = np.array([s[0,0] for s in seg_val])


    ### hrlimits are derived from the data, in the GRS1915_DataVisualisation Notebook
    hrlimits = [[-2.5, 1.5], [-3.0, 2.0]]


    features_train = make_features(seg_train,k, bins, lamb, hr_summary, ps_summary, lc, hr, hrlimits=hrlimits)
    features_test = make_features(seg_test,k, bins, lamb,  hr_summary, ps_summary, lc, hr, hrlimits=hrlimits)

    print("len(tstart_train): " + str(len(tstart_train)))
    print("len(features_train): " + str(len(features_train)))

    features_train["tstart"] = tstart_train
    features_test["tstart"] = tstart_test

    features_train["nseg"] = nseg_train
    features_test["nseg"] = nseg_test
    #features_train = np.concatenate((tstart_train, features_train))
    #features_test = np.concatenate((tstart_test, features_test))

    print("len(tstart_test): " + str(len(tstart_test)))
    print("len(features_test): " + str(len(features_test)))


    ## check for NaN
    print("Checking for NaN in the training set ...")
    features_train_checked, labels_train_checked = check_nan(features_train, labels_train,
                                                             hr=hr, lc=lc)
    print("Checking for NaN in the test set ...")
    features_test_checked, labels_test_checked= check_nan(features_test, labels_test,
                                                          hr=hr, lc=lc)


    labelled_features = {"train": [features_train_checked["features"], labels_train_checked],
                     "test": [features_test_checked["features"], labels_test_checked]}

    if val:
        features_val = make_features(seg_val, k, bins, lamb, hr_summary, ps_summary, lc, hr, hrlimits=hrlimits)
        #features_val = np.concatenate((tstart_val, features_val))
        features_val["tstart"] = tstart_val
        features_val["nseg"] = nseg_val

        features_val_checked, labels_val_checked = check_nan(features_val, labels_val, hr=hr, lc=lc)

        labelled_features["val"] =  [features_val_checked["features"], labels_val_checked],

    print("Checking for NaN in the validation set ...")

    if save_features:
        np.savetxt(froot+"_%is_features_train.txt"%int(seg_length), features_train_checked["features"])
        np.savetxt(froot+"_%is_features_test.txt"%int(seg_length), features_test_checked["features"])
        np.savetxt(froot+"_%is_nseg_train.txt"%int(seg_length), features_train_checked["nseg"])
        np.savetxt(froot+"_%is_nseg_test.txt"%int(seg_length), features_test_checked["nseg"])

        np.savetxt(froot+"_%is_tstart_train.txt"%int(seg_length), features_train_checked["tstart"])
        np.savetxt(froot+"_%is_tstart_test.txt"%int(seg_length), features_test_checked["tstart"])

        ltrainfile = open(froot+"_%is_labels_train.txt"%int(seg_length), "w")
        for l in labels_train_checked:
            ltrainfile.write(str(l) + "\n")
        ltrainfile.close()

        ltestfile = open(froot+"_%is_labels_test.txt"%int(seg_length), "w")
        for l in labels_test_checked:
            ltestfile.write(str(l) + "\n")
        ltestfile.close()


        if val:
            np.savetxt(froot+"_%is_features_val.txt"%int(seg_length), features_val_checked["features"])
            np.savetxt(froot+"_%is_tstart_val.txt"%int(seg_length), features_val_checked["tstart"])
            np.savetxt(froot+"_%is_nseg_val.txt"%int(seg_length), features_val_checked["nseg"])

            lvalfile = open(froot+"_%is_labels_val.txt"%int(seg_length), "w")
            for l in labels_val_checked:
                lvalfile.write(str(l) + "\n")
            lvalfile.close()


        if lc:
            lc_all = {"train":features_train_checked["lc"], "test":features_test_checked["lc"]}
            if val:
                lc_all["val"] = features_val_checked["lc"]

            f = open(froot+"_%is_lc_all.dat"%int(seg_length), "w")
            pickle.dump(lc_all, f, -1)
            f.close()

        if hr:
            hr_all = {"train":features_train_checked["hr"], "test":features_test_checked["hr"]}
            if val:
                hr_all["val"] = features_val_checked["hr"]

            f = open(froot+"_%is_hr_all.dat"%int(seg_length), "w")
            pickle.dump(hr_all, f, -1)
            f.close()

        #f = open(froot+"_features.dat", "w")
        #pickle.dump(labelled_features, f)
        #f.close()

    return labelled_features



######################################################################################################################
#### EXTRACT FEATURE FILES ###########################################################################################
######################################################################################################################

def extract_all(d_all, seg_length_all=[256., 1024.], overlap=128.,
                val=True, train_frac=0.5, validation_frac = 0.25, test_frac = 0.25,
                k = 10, lamb=0.1,
                datadir="./"):

    if np.size(overlap) != np.size(seg_length_all):
        overlap = [overlap for i in xrange(len(seg_length_all))]

    print(overlap)
    print("seg_length: " + str(seg_length_all))
    print("overlap: " + str(overlap))

    for ov, sl in zip(overlap, seg_length_all):
        print("%i segments, summary"%int(sl))
        lf = make_all_features(d_all, k, lamb, val, train_frac, validation_frac, test_frac,
                  seg=True, seg_length=sl, overlap=ov,
                  hr_summary=True, ps_summary=True, lc=True, hr=True,
                  save_features=True, froot=datadir+"grs1915")

    return
