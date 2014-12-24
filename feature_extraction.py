
import matplotlib.pyplot as plt
import numpy as np
import glob
import cPickle as pickle
import pandas as pd
import astroML.stats
import scipy.stats

import lightcurve
import powerspectrum



def get_labelled_data(filename, seg_length=256., overlap=64., tres = 0.125):
 
    f = open(filename)
    d_all = pickle.load(f)
    f.close()
 
    print("Number of light curves:  " + str(len(d_all)))
    states = [d[1] for d in d_all]
    st = pd.Series(states)
    st.value_counts()
 
    ## total number of light curves
    n_lcs = len(d_all)
    
    ## shuffle list of light curves
    np.random.shuffle(d_all)

    train_frac = 0.5
    validation_frac = 0.25
    test_frac = 0.25

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

    if seg_length is not None:
        seg_train, labels_train = extract_segments(d_all_train, seg_length=seg_length, overlap=overlap)
        seg_val, labels_val = extract_segments(d_all_val, seg_length=seg_length, overlap=overlap)
        seg_test, labels_test = extract_segments(d_all_test, seg_length=seg_length, overlap=overlap)
    else:
        seg_train = [d[0] for d in d_all_train] 
        seg_val = [d[0] for d in d_all_val]
        seg_test = [d[0] for d in d_all_test]
        labels_train = [d[1] for d in d_all_train]
        labels_val = [d[1] for d in d_all_val]
        labels_test = [d[1] for d in d_all_test]

    ## Let's print some details on the different segment data sets
    print("There are %i segments in the training set."%len(seg_train))
    print("There are %i segments in the validation set."%len(seg_val))
    print("There are %i segments in the test set."%len(seg_test))
    for la,n in zip([labels_train, labels_val, labels_test], ["training", "validation", "test"]):
        print("These is the distribution of states in the %s set: "%n)
        st = pd.Series(la)
        print(st.value_counts())
        print("================================================================")

    return [seg_train, labels_train], [seg_val, labels_val], [seg_test, labels_test]







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
    ps_new = ps[:nbins_new*int(n)]
    binps = np.reshape(np.array(ps_new), (nbins_new, n))
    binps = np.sum(binps, axis=1)
    if type in ["average", "mean"]:
        binps = binps/np.float(n)
    else:
        binps = binps

    if len(binfreq) < len(binps):
        binps= binps[:len(binfreq)]

    return binfreq, binps

def timeseries_features(seg):
    times = seg[:,0]
    counts = seg[:,1]
    fmean = np.mean(counts)
    fmedian = np.median(counts)
    fvar = np.var(counts)
    return fmean, fmedian, fvar


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
    if times[-1]-times[0] >= 2.*256.:
        tlen = (times[-1]-times[0])
        nrebin = np.round(tlen/256.)
        freq, ps = rebin_psd(freq, ps, n=nrebin, type='average')
    
    #print("len(ps), after: " + str(len(ps)))

    freq = freq[1:]
    ps = ps[1:]

    fmax_ind = np.where(ps == np.max(ps))[0]
    maxfreq = freq[fmax_ind[0]]

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

def total_psd(seg, bins=30):
    times = seg[:,0]    
    dt = times[1:] - times[:-1]
    dt = np.min(dt)
    #print("tseg: " + str(times[-1] - times[0]))
    counts = seg[:,1]*dt
    
    lc = lightcurve.Lightcurve(times, counts=counts, timestep = dt, tseg=np.ceil(times[-1]-times[0]), tstart=times[0])
    ps = powerspectrum.PowerSpectrum(lc, norm="rms", )
    #print("log(ps): " + str(np.log10(ps.freq[-1]) - np.log10(ps.freq[1])))

    binfreq = np.logspace(np.log10(ps.freq[1]*0.9), np.log10(ps.freq[-1]), bins)
    binps, bin_edges, binno = scipy.stats.binned_statistic(ps.freq[1:], ps.ps[1:], statistic="mean", bins=binfreq)
    if any(np.isnan(binps)):
      print("freq: " + str(ps.freq[1:10])) 
      print("bins: " + str(binfreq[:10]))
      print("bin_edges: " + str(bin_edges[:10]))
      print("NaN! dt: %.3f,tseg: %.3f, ps: %.3f"%(dt,(times[-1] - times[0]), (np.log10(ps.freq[-1]) - np.log10(ps.freq[1]))))
    return binps




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
        return ps.freq, ps.ps
    else:
        ps_all = []
        for n in xrange(navg):
            t_small = times[n*nseg:(n+1)*nseg]
            c_small = counts[n*nseg:(n+1)*nseg]
            ps = powerspectrum.PowerSpectrum(t_small, counts=c_small, norm="rms")
            ps_all.append(ps.ps)
        
        #print(np.array(ps_all).shape) 
    
        ps_all = np.average(np.array(ps_all), axis=0)

        #print(ps_all.shape) 
    
        return ps.freq, ps_all

hr_limits = [[0.292, 1.],[0.00, 0.708]]
hid_limits = [[0.1, 1.5], [0.0, 6000.0]]


def hr_maps(seg, hr_limits, bins=30):
    times = seg[:,0]
    counts = seg[:,1]
    low_counts = seg[:,2]
    high_counts = seg[:,3]
    hr1 = low_counts/counts
    hr2 = high_counts/counts
    h, xedges, yedges = np.histogram2d(hr1, hr2, bins=bins, 
                                       range=hr_limits)
    h = np.rot90(h)
    h = np.flipud(h)
    
    return xedges, yedges, h

def hid_maps(seg, hid_limits, bins=30):
    times = seg[:,0]
    counts = seg[:,1]
    low_counts = seg[:,2]
    high_counts = seg[:,3]
    hr2 = high_counts/low_counts
    h, xedges, yedges = np.histogram2d(hr2, counts, bins=bins, 
                                       range=hid_limits)
    h = np.rot90(h)
    h = np.flipud(h)
    
    return xedges, yedges, h


def hr_fitting(seg):
    times = seg[:,0]
    counts = seg[:,1]
    low_counts = seg[:,2]
    high_counts = seg[:,3]
    hr1 = low_counts/counts
    hr2 = high_counts/counts

    # compute the robust statistics
    #(mu_r, sigma1_r, sigma2_r, alpha_r) = astroML.stats.fit_bivariate_normal(hr1, hr2, robust=False)

    #return mu_r, sigma1_r, sigma2_r, alpha_r
    mu1 = np.mean(hr1)
    mu2 = np.mean(hr2)
    #cov = np.cov(hr1, hr2)
    #return mu1, mu2, cov.flatten()
    sigma1 = np.var(hr1)
    sigma2 = np.var(hr2)
    return mu1, mu2, sigma1, sigma2


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


def make_features(seg, bins=30, navg=4, lc=True, hr=True):
    features = []
    if lc:
        lc_all = []
    if hr:
        hr_all = []
    for s in seg:
        features_temp = []
        fmean, fmedian, fvar = timeseries_features(s)
        features_temp.extend([fmean, fmedian, fvar])

        maxfreq, psd_a, psd_b, psd_c, psd_d, pc1, pc2 = psd_features(s, pcb)
        binps = total_psd(s, 10)
	features_temp.extend(binps)         
        #features_temp.extend([maxfreq, psd_a, psd_b, psd_c, psd_d, pc1, pc2])
	#whole_ps = total_psd(s,navg=navg)
        #features_temp.extend(whole_ps[1:])

        #lc = lcshape_features(s, dt=1.0)
        #features_temp.extend(lc)
         
        #xedges, yedges, h = hr_maps(s, hr_limits, bins=bins)
        #features_temp.extend(h.flatten())
	 
	xedges, yedges, h = hid_maps(s, hid_limits, bins=bins)
	features_temp.extend(h.flatten())
	#mu, sigma1, sigma2, alpha = hr_fitting(s)
        #features_temp.extend([mu[0], mu[1], sigma1, sigma2, alpha])
        #mu1, mu2, sigma1, sigma2 = hr_fitting(s)
        #features_temp.extend([mu1, mu2, sigma1, sigma2])
	#mu1, mu2, cov = hr_fitting(s)
	#features_temp.extend([mu1, mu2])
	#features_temp.extend(cov)

        features.append(features_temp)
        
        if lc or hr:
            lc_temp = extract_lc(s)
        if lc:
            #print("appending light curve")
            lc_all.append([lc_temp[0], lc_temp[1]])
        if hr:
            #print("appending hardness ratios")
            hr_all.append([lc_temp[2], lc_temp[3]])
            
    print("I am about to make a dictionary")
    fdict = {"features": features}
    print(fdict.keys)
    if lc:
        print("I am in lc")
        #features.append(lc_all)
        fdict["lc"] = lc_all
    if hr:
        print("I am in hr")
        #features.append(hr_all)
        fdict["hr"] = hr_all
    print(fdict.keys())
    return fdict




