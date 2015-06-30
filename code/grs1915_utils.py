### Random utility function for GRS 1915 Analysis Stuff
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.cm as cmap

import seaborn as sns

import numpy as np
import pandas as pd
import cPickle as pickle

from collections import Counter


def extract_segments(d_all, seg_length = 256., overlap=64.):
    """ Extract light curve segmens from a list of light curves. 
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
    segments, labels = [], [] ## labels for labelled data
        
    for i,d_seg in enumerate(d_all):
        
        ## data is an array in the first element of d_seg
        data = d_seg[0]
        ## state is a string in the second element of d_seg
        state = d_seg[1]

        ## compute the intervals between adjacent time bins
        dt_data = data[1:,0] - data[:-1,0]
        dt = np.min(dt_data)
        #print("dt: " + str(dt))
        
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
        
    return segments, labels


def state_distribution(labels_all, seg, namestr="test", datadir="./"):
    """
    Plot the overall distribution of states.

    Parameters
    ----------
    labels_all: {list, ndarray}
        list of all labels

    seg: float
        The length of the segments used for the classification

    namestr: string, optional, default "test"
        string with the output file name

    datadir: string, optional, default "./"
        Location of the output matplotlib figure


    """


    s = pd.Series(labels_all)
    nstates = s.value_counts()


    nstates.plot(kind='bar')
    plt.title("Overall distribution of classified states", fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.ylabel("Number of %i-second segments in state"%int(seg), fontsize=15)
    plt.savefig(datadir+namestr+".pdf", format="pdf")
    plt.close()


    nstates *= seg

    nstates.plot(kind='bar')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.title("Overall distribution of  states", fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.ylabel("Seconds spent in state"%int(seg), fontsize=15)
    plt.savefig(datadir+namestr+"_seconds.pdf", format="pdf")
    plt.close()

    return


def state_time_evolution(times, labels, namestr="test", datadir="./"):
    """
    How do the states evolve with time?
    Plot overall ASM light curve and mark the places
    where observations occurred, colour-coded by state.


    Parameters
    -----------
    times: {list, ndarray}
        list with times in MET seconds

    labels: {list, ndarray}
        list or array with state labels
    """

    ## The MJD reference time
    mjdrefi = 49353.

    ## convert all times from MET to MJD
    times /= (60*60*24.)
    times += mjdrefi

    ## convert string labels to numbers
    label_set = np.unique(labels)
    labels_numeric = np.array([np.where(l == label_set)[0][0] for l in labels])


    ## load and plot ASM light curve
    asm = np.loadtxt(datadir+"grs1915_asm_lc.txt",skiprows=5)
    asm_time = asm[:,0]
    asm_cr = asm[:,1]
    asm_total = asm_time[-1]-asm_time[0]
    print("The ASM light curve covers a total of %i days"%asm_total)

    ## each light curve covers 500 days
    plot_len = 500.
    start_time = asm_time[0]
    end_time = start_time + plot_len

    print("min(times): " + str(np.min(times)))
    print("asm time zero: " + str(asm_time[0]))

    ## now make the actual figure!
    fig = plt.figure(figsize=(12,15))
    gs = gridspec.GridSpec(11, 5)

    sns.set_style("white")

    ax = plt.subplot(gs[:,:])
    # Turn off axis lines and ticks of the big subplot

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    sns.set_context("notebook", font_scale=1.0, rc={"axes.labelsize": 16})

    sns.set_style("darkgrid")

    plt.rc("font", size=16, family="serif", serif="Computer Sans")
    plt.rc("axes", titlesize=16, labelsize=16)
    plt.rc("text", usetex=True)

    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.97, hspace=0.2)

    current_palette = sns.color_palette(n_colors=len(label_set))
    colours = [current_palette[j] for j in labels_numeric]


    i = 0
    while end_time <= asm_time[-1]:
        print("I am on plot %i."%i)
        #ax1 = fig.add_subplot(11,1,i+1)
        ax1 = plt.subplot(gs[i, :4])

        ax1.errorbar(asm[:,0], asm[:,1], yerr = asm[:,2], linestyle="steps-mid")
        path = ax1.scatter(times, np.ones(len(times))*240., facecolor=colours,
                    edgecolor="None")
        ax1.set_xlim([start_time, end_time])
        ax1.set_ylim([1.0, 299.0])
        ax1.set_yticks(np.arange(3)*100.0+100.0, [100, 200, 300]);

        start_time +=plot_len
        end_time += plot_len
        i+=1



    ### HORRIBLY COMPLICATED WAY TO MAKE THE LEGEND
    lines = []

    sns.set_style("white")
    ax2 = plt.subplot(gs[:3,4])

    # Turn off axis lines and ticks of the big subplot
    ax2.spines['top'].set_color('none')
    ax2.spines['bottom'].set_color('none')
    ax2.spines['left'].set_color('none')
    ax2.spines['right'].set_color('none')
    ax2.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    for i,l in enumerate(label_set):
        line = mlines.Line2D([], [], color=current_palette[i], marker='o',
                              markersize=9, label=l, linewidth=0, zorder=10)
        lines.append(line)

    ax2.legend(handles=lines, loc='upper left', shadow=True)

    ### LABELS
    ax.set_xlabel("Time in MJD", fontsize=18)
    ax.set_ylabel("Count rate [counts/s]", fontsize=18)

    plt.savefig(datadir+namestr+"asm_all.pdf", format="pdf")
    plt.close()


    return


def plot_classified_lightcurves(times, labels, namestr="test", datadir="./"):
    f = open(datadir+"grs1915_all_125ms.dat")
    d_all = pickle.load(f)
    f.close()

    ## total number of light curves
    n_lcs = len(d_all)

    ## Set the seed to I will always pick out the same light curves.
    np.random.seed(20150608)

    ## shuffle list of light curves
    np.random.shuffle(d_all)

    train_frac = 0.6
    validation_frac = 0.2
    test_frac = 0.2

    ## let's pull out light curves for three data sets into different variables.
    d_all_train = d_all[:int(train_frac*n_lcs)]
    d_all_val = d_all[int(train_frac*n_lcs):int((train_frac + validation_frac)*n_lcs)]
    d_all_test = d_all[int((train_frac + validation_frac)*n_lcs):]

    colors = np.loadtxt("colors.txt")

    clist = [colors[i] for i in labels]
    for i,d in enumerate(d_all_train[:10]):
        data = d[0]
        times = data[:,0]
        counts = data[:,1]
        plt.figure()
        plt.plot(times, counts, lw=1, linestyle="steps-mid")
        mincounts = np.min(counts)
        maxcounts = np.max(counts)
        plt.axis([times[0], times[-1], mincounts-0.1*mincounts, maxcounts+0.1*maxcounts])
        plt.scatter(fscaled_all[:,0], np.ones_like(tstart_all)*1.05*maxcounts, color=clist)
        plt.savefig("../../grs1915_lc%i_meanshift.pdf"%i, format="pdf")
        plt.close()



    return


def transition_matrix(times, labels, gapsize, namestr="test", datadir="./"):
    """
    Make a transition matrix between states. Remove all transitions between
    time points in array time that are more than a gapsize apart (these are
    breaks in the observations).

    Parameters
    ----------
    times: {list, ndarray}
        List of observation times

    labels: {list, ndarray}
        List of state labels to go with the observation times

    gapsize: float
        The maximum distance between two elements in `times`, where both
        elements would still be considered connected.

    namestr: string, optional, default "test"
        String object describing the data to be used at part of the filename

    datadir: string, optional, default "./"
        Directory where the data is located, and where the output plot will
        be put. Default is the current working directory.
    """

    ## convert labels from strings to numbers
    label_set = np.unique(labels)
    labels_numeric = np.array([np.where(l == label_set)[0][0] for l in labels])


    ## stack labels and start times
    time_labels = np.vstack((times, labels_numeric))

    ## sort by observation times
    tl_sorted = np.array(sorted(time_labels.T, key=lambda a_entry: a_entry[0]))


    ## compute the difference between starting points of consecutive
    ## segments in seconds
    dt = np.diff(tl_sorted[:,0])

    ## find all indices where segments are more than 1024 seconds apart
    breakind = np.where(dt > gapsize)[0]

    sorted_times = tl_sorted[:,0]
    sorted_states = tl_sorted[:,1]

    sorted_states_new = [sorted_states[:breakind[0]]]
    for i in xrange(len(breakind[:-1])):
        sorted_states_new.append(sorted_states[breakind[i]:breakind[i+1]])


    ## number of unique states
    nlabels = len(label_set)


    ## compute transitions between neighbouring states
    ## for all connected observation times
    b_all = []
    for s in sorted_states_new:
        b = np.zeros((nlabels,nlabels))

        for (x,y), c in Counter(zip(s, s[1:])).iteritems():
            b[x-1,y-1] = c

        b_all.append(b)


    b_all = np.array(b_all)
    b = np.sum(b_all, axis=0)

    ## normalize to make each row sum up to 1
    b = np.array([x/xsum for x,xsum in zip(b,np.sum(b, axis=1))])
    #b /= np.sum(b, axis=0)

    plt.figure()
    plt.matshow(b, cmap=cmap.Spectral_r)
    plt.xlabel("Transition to state")
    plt.ylabel("Transition from state")

    plt.savefig(datadir+namestr+"_transitionmatrix.pdf", format="pdf")
    plt.close()

    return