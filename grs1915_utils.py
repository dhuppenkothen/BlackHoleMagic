### Random utility function for GRS 1915 Analysis Stuff

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
    
    


