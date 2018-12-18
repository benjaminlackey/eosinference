import h5py
import pandas as pd

import numpy as np
import scipy.stats as stats
import scipy.interpolate as interpolate

import bounded_2d_kde


################################################################################
# Functions for converting (q, lambdat) samples to a function ln(p(q, lambdat))#
################################################################################

def construct_p_of_ql_bounded_kde(qs, lambdats, kde_bound_limits, bw_method=None, log=False):
    """Construct a function that approximates the pseudolikelihood
    ln(p(q, lambdat)) from the MCMC samples for a BNS event using
    a 2d bounded KDE.

    Parameters
    ----------
    qs : array of q MCMC samples
    lambdats : array of lambdat MCMC samples

    Returns
    -------
    lnp_of_ql_kde : function(q, lambdat)
    """
    # Initialize KDE
    data = np.array([qs, lambdats]).T
    xlow, xhigh, ylow, yhigh = kde_bound_limits
    kde = bounded_2d_kde.Bounded_2d_kde(
        data,
        xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh,
        bw_method=bw_method)

    # Return a function that takes 2 arguments instead of a
    # function that takes the array [q, lambdat]
    def pseudolike(q, lambdat, log=False):
        if log==False:
            return kde.evaluate(np.array([q, lambdat]))[0]
        else:
            # TODO: This method doesn't work yet
            return kde.logpdf(np.array([q, lambdat]))[0]

    return pseudolike


def construct_p_of_ql_gaussian(qs, lambdats):
    """Construct a function that approximates the pseudolikelihood
    ln(p(q, lambdat)) as a mutivariate Gaussian using the
    MCMC samples for a BNS event. This is done to extend the KDE
    far from the region where there MCMC samples.

    Parameters
    ----------
    qs : array of q MCMC samples
    lambdats : array of lambdat MCMC samples

    Returns
    -------
    lnp_of_ql_gaussian : function(q, lambdat)
    """
    data = np.array([qs, lambdats])
    mu = np.mean(data, axis=1)
    sigma = np.cov(data)
    distribution = stats.multivariate_normal(mean=mu, cov=sigma)

    # Return a function that takes 2 arguments instead of a
    # function that takes the array [q, lambdat]
    def pseudolike(q, lambdat, log=False):
        if log==False:
            return distribution.pdf(np.array([q, lambdat]))
        else:
            return distribution.logpdf(np.array([q, lambdat]))

    return pseudolike


def construct_lnp_of_ql_grid(qs, lambdats, kde_bound_limits, grid_limits,
                             gridsize=500, bw_method=None):
    """
    """
    # Construct the approximates for the pseudolikelihood
    p_of_ql_kde = construct_p_of_ql_bounded_kde(qs, lambdats, kde_bound_limits, bw_method=bw_method)
    p_of_ql_gaussian = construct_p_of_ql_gaussian(qs, lambdats)

    # Make a list of [x, y] coordinates for the grid
    q_grid = np.linspace(grid_limits[0], grid_limits[1], gridsize)
    l_grid = np.linspace(grid_limits[2], grid_limits[3], gridsize)

    lnp_grid = np.array([[[q, l, p_of_ql_gaussian(q, l, log=True)] for l in l_grid] for q in q_grid])
    lnp_max = np.max(lnp_grid[:, :, 2])
    for i in range(len(q_grid)):
        for j in range(len(l_grid)):
            lnp = lnp_grid[i, j, 2]
            # Replace with KDE if less than 20 e-folds from max value
            if lnp > lnp_max-40.0:
                q = lnp_grid[i, j, 0]
                l = lnp_grid[i, j, 1]
                lnp_grid[i, j, 2] = np.log(p_of_ql_kde(q, l))
    return lnp_grid


def pseudolikelihood_data_from_pe_samples(
    filename,
    kde_bound_limits=[0.5, 1.0, 0.0, 5000.0],
    grid_limits=[0.5, 1.0, 0.0, 5000.0],
    gridsize=250):
    """Open a CSV data file for a BNS MCMC run and evaluate
    ln(pseudolikelihood(q, lambdat)) on a grid.

    Parameters
    ----------
    filename : CSV file
        MCMC samples with column headers named ['mc', 'q', 'lambdat']

    Returns
    -------
    mc_mean : mean value of chirp mass
    lnp_of_ql_grid : 3d array
        [q, lambdat, lnp] for each value of q and lambdat
    """
    # Open csv file as pandas data frame
    df = pd.read_csv(filename)

    # Get chirp mass mean
    mc_mean = df['mc'].mean()

    # Construct grid that describes lnp(q, lambdat)
    qs = df['q'].values
    lambdats = df['lambdat'].values

    lnp_of_ql_grid = construct_lnp_of_ql_grid(
        qs, lambdats, kde_bound_limits, grid_limits,
        gridsize=gridsize, bw_method=None)

    return mc_mean, lnp_of_ql_grid


################################################################################
#      Interpolate the gridded function lnp(q, lambdat)                        #
################################################################################

def interpolate_lnp_of_ql_from_grid(lnp_of_ql_grid):
    """Interpolate the function ln[p(q, lam_tilde)] on a grid of q, lam_tilde values.
    """
    q_grid = lnp_of_ql_grid[:, 0, 0]
    l_grid = lnp_of_ql_grid[0, :, 1]
    lnp_grid = lnp_of_ql_grid[:, :, 2]
    lnp_of_ql = interpolate.RectBivariateSpline(
        q_grid, l_grid, lnp_grid,
        bbox=[None, None, None, None], kx=2, ky=2, s=0)
    return lnp_of_ql.ev


################################################################################
#      Save and load the gridded distributions lnp(q, lambdat)                 #
################################################################################

def save_pseudolikelihood_data(filename, mc_mean_list, lnp_of_ql_grid_list):
    """Save the data needed to construct the pseudolikelihoods
    for each of the nbns systems.
    """
    f = h5py.File(filename)
    nbns = len(mc_mean_list)
    for i in range(nbns):
        groupname = 'bns_{}'.format(i)
        group = f.create_group(groupname)
        group.attrs['mc_mean'] = mc_mean_list[i]
        group['lnp_of_ql_grid'] = lnp_of_ql_grid_list[i]
    f.close()


def load_pseudolikelihood_data(filename):
    """Load the data needed to construct the pseudolikelihoods
    for each of the nbns systems.
    """
    f = h5py.File(filename)
    mc_mean_list = []
    lnp_of_ql_grid_list = []
    nbns = len(f.keys())
    for i in range(nbns):
        # Get data
        groupname = 'bns_{}'.format(i)
        mc = f[groupname].attrs['mc_mean']
        lnp = f[groupname]['lnp_of_ql_grid'][:]
        # add data to lists
        mc_mean_list.append(mc)
        lnp_of_ql_grid_list.append(lnp)
    f.close()
    return mc_mean_list, lnp_of_ql_grid_list
