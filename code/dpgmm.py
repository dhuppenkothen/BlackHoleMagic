
import numpy as np

import plotting
import feature_engineering
import multiprocessing as mp

import cPickle as pickle

from microscopes.common.rng import rng
from microscopes.common.recarray.dataview import numpy_dataview
from microscopes.models import niw as normal_inverse_wishart
from microscopes.mixture.definition import model_definition
from microscopes.mixture import model, runner, query
from microscopes.common.query import zmatrix_heuristic_block_ordering, zmatrix_reorder
from microscopes.kernels import parallel


from paper_figures import load_data


def run_dpgmm(niter=1000, datadir="../../", nfeatures=13):

    ranking = [10,  6,  7, 26,  5,  8,  4, 19, 12, 23, 24, 33, 28, 25,
               14,  3,  0, 1, 21, 30, 11, 31, 13,  9, 22,  2, 27, 29,
               32, 17, 18, 20, 16, 15]

    features, labels, lc, hr, tstart, \
        features_lb, labels_lb, lc_lb, hr_lb, \
        fscaled, fscaled_lb, fscaled_full, labels_all = \
            load_data(datadir, tseg=1024.0, log_features=None,
                      ranking=ranking)

    labels_phys = feature_engineering.convert_labels_to_physical(labels)
    labels_phys_lb = feature_engineering.convert_labels_to_physical(labels_lb)

    labels_all_phys = np.hstack([labels_phys["train"], labels_phys["val"],
                                 labels_phys["test"]])


    fscaled_small = fscaled_full[:, :13]

    nchains = 8

    # The random state object
    prng = rng()

    # Define a DP-GMM where the Gaussian is 2D
    defn = model_definition(fscaled_small.shape[0],
                            [normal_inverse_wishart(fscaled_small.shape[1])])

    fscaled_rec = np.array([(list(f),) for f in fscaled_small],
                           dtype=[('', np.float32, fscaled_small.shape[1])])

    # Create a wrapper around the numpy recarray which
    # data-microscopes understands
    view = numpy_dataview(fscaled_rec)

    # Initialize nchains start points randomly in the state space
    latents = [model.initialize(defn, view, prng) for _ in xrange(nchains)]

    # Create a runner for each chain
    runners = [runner.runner(defn, view, latent,
                             kernel_config=['assign']) for latent in latents]
    r = parallel.runner(runners)

    r.run(r=prng, niters=niter)

    with open(datadir+"grs1915_dpgmm.pkl", "w") as f:
        pickle.dump(r, f)

    return

if __name__ == "__main__":
    run_dpgmm(50000)