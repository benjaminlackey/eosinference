"""This module provides methods to construct a function that approximates the
pseudolikelihood ln[p(q, lambdat)] from the MCMC samples for a BNS event. The
density is estimated with a 2d bounded KDE. For fast evaluation, it is then
gridded up and interpolated. Functions are provided to save and load the gridded
data.
"""

import h5py
import pandas as pd

import numpy as np
import scipy.stats as stats
import scipy.interpolate as interpolate

import bounded_2d_kde


################################################################################
# Functions for converting (q, lambdat) samples to a function ln(p(q, lambdat))#
################################################################################

def construct_p_of_ql_bounded_kde(qs, lambdats, kde_bound_limits, bw_method=None):
    """Construct a function that approximates the pseudolikelihood
    p(q, lambdat) from the MCMC samples for a BNS event using a bounded 2d KDE.

    Parameters
    ----------
    qs : 1d array of q MCMC samples
    lambdats : 1d array of lambdat MCMC samples
    kde_bound_limits : [qlow, qhigh, lambdatlow, lambdathigh]

    Returns
    -------
    p_of_ql : function(q, lambdat)
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
    def p_of_ql(q, lambdat):
        return kde.evaluate(np.array([q, lambdat]))[0]

    return p_of_ql


def construct_lnp_of_ql_gaussian_kde(qs, lambdats, bw_method=None):
    """Construct a function that approximates the ln[pseudolikelihood]
    ln[p(q, lambdat)] from the MCMC samples for a BNS event using a Gaussian KDE.

    Parameters
    ----------
    qs : 1d array of q MCMC samples
    lambdats : 1d array of lambdat MCMC samples

    Returns
    -------
    lnp_of_ql : function(q, lambdat)
    """
    # Initialize KDE
    kde = stats.gaussian_kde(np.array([qs, lambdats]), bw_method=bw_method)

    # Return a function that takes 2 arguments instead of a
    # function that takes the array [q, lambdat]
    def lnp_of_ql(q, lambdat):
            return kde.logpdf(np.array([q, lambdat]))[0]

    return lnp_of_ql


# def construct_p_of_ql_gaussian(qs, lambdats):
#     """Construct a function that approximates the pseudolikelihood
#     ln(p(q, lambdat)) as a mutivariate Gaussian using the
#     MCMC samples for a BNS event. This is done to extend the KDE
#     far from the region where there are no MCMC samples and the KDE
#     suffers from underflow
#
#     Parameters
#     ----------
#     qs : array of q MCMC samples
#     lambdats : array of lambdat MCMC samples
#
#     Returns
#     -------
#     lnp_of_ql_gaussian : function(q, lambdat)
#     """
#     data = np.array([qs, lambdats])
#     mu = np.mean(data, axis=1)
#     sigma = np.cov(data)
#     distribution = stats.multivariate_normal(mean=mu, cov=sigma)
#
#     # Return a function that takes 2 arguments instead of a
#     # function that takes the array [q, lambdat]
#     def pseudolike(q, lambdat, log=False):
#         if log==False:
#             return distribution.pdf(np.array([q, lambdat]))
#         else:
#             return distribution.logpdf(np.array([q, lambdat]))
#
#     return pseudolike


################################################################################
#    Function for evaluating lnp(q, lambdat) on a grid                         #
################################################################################

def construct_lnp_of_ql_grid(qs, lambdats, kde_bound_limits, grid_limits,
                             gridsize=500, bw_method=None):
    """Grid up the KDE approximation of ln(p(q, lambdat)). If the bounded KDE
    suffers from underflow, then use the Gaussian KDE instead which has as
    logpdf method that is well behaved.

    Parameters
    ----------
    qs : 1d array of q MCMC samples
    lambdats : 1d array of lambdat MCMC samples
    kde_bound_limits : [qlow, qhigh, lambdatlow, lambdathigh]
    gridsize : Number of points to sample in the q and lambdat directions.
    """
    # Construct the approximations for the pseudolikelihood
    p_of_ql_bounded_kde = construct_p_of_ql_bounded_kde(qs, lambdats, kde_bound_limits, bw_method=bw_method)
    lnp_of_ql_gaussian_kde = construct_lnp_of_ql_gaussian_kde(qs, lambdats, bw_method=bw_method)

    # Make a list of [x, y] coordinates for the grid
    q_grid = np.linspace(grid_limits[0], grid_limits[1], gridsize)
    l_grid = np.linspace(grid_limits[2], grid_limits[3], gridsize)

    # Allocate memory
    lnp_grid = np.array([[[q, l, 0.0] for l in l_grid] for q in q_grid])

    # Fill the lnp value with the KDE approximation.
    # If there is underflow, use the Gaussian approximation instead.
    for i in range(len(q_grid)):
        for j in range(len(l_grid)):
            q = lnp_grid[i, j, 0]
            l = lnp_grid[i, j, 1]
            p = p_of_ql_bounded_kde(q, l)
            if p<1.0e-300:
                # Rough threshold for when np.log(p) returns -inf
                lnp = lnp_of_ql_gaussian_kde(q, l)
            else:
                lnp = np.log(p)
            lnp_grid[i, j, 2] = lnp
    return lnp_grid

# def construct_lnp_of_ql_grid(qs, lambdats, kde_bound_limits, grid_limits,
#                              gridsize=500, bw_method=None):
#     """Grid up the KDE approximation of ln(p(q, lambdat)). If the KDE
#     suffers from underflow, then use the Gaussian approximation instead.
#
#     Parameters
#     ----------
#     qs : 1d array of q MCMC samples
#     lambdats : 1d array of lambdat MCMC samples
#     kde_bound_limits : [qlow, qhigh, lambdatlow, lambdathigh]
#     gridsize : Number of points to sample in the q and lambdat directions.
#     """
#     # Construct the approximations for the pseudolikelihood
#     p_of_ql_kde = construct_p_of_ql_bounded_kde(qs, lambdats, kde_bound_limits, bw_method=bw_method)
#     p_of_ql_gaussian = construct_p_of_ql_gaussian(qs, lambdats)
#
#     # Make a list of [x, y] coordinates for the grid
#     q_grid = np.linspace(grid_limits[0], grid_limits[1], gridsize)
#     l_grid = np.linspace(grid_limits[2], grid_limits[3], gridsize)
#
#     # Allocate memory
#     lnp_grid = np.array([[[q, l, 0.0] for l in l_grid] for q in q_grid])
#
#     # Fill the lnp value with the KDE approximation.
#     # If there is underflow, use the Gaussian approximation instead.
#     for i in range(len(q_grid)):
#         for j in range(len(l_grid)):
#             q = lnp_grid[i, j, 0]
#             l = lnp_grid[i, j, 1]
#             p = p_of_ql_kde(q, l)
#             if p<1.0e-300:
#                 # Rough threshold for when np.log(p) returns -inf
#                 lnp = p_of_ql_gaussian(q, l, log=True)
#             else:
#                 lnp = np.log(p)
#             lnp_grid[i, j, 2] = lnp
#     return lnp_grid


################################################################################
#    Function for interpolating lnp(q, lambdat)                                #
################################################################################

def interpolate_lnp_of_ql_from_grid(lnp_of_ql_grid):
    """Interpolate the function ln[p(q, lam_tilde)] from a grid, where
    lnp is the log likelihood for a given BNS system. This interpolated function
    is used instead of the original KDE to allow fast evaluation.

    Parameters
    ----------
    lnp_of_ql_grid : 3d-array of shape (nq, nlambdat, 3)
        A 2d grid of points that contain the values [q, lambdat, lnp].

    Returns
    -------
    func(q, lambdat) : 2nd order polynomial interpolating function
        Takes the arguments q, lambdat. Returns a float.
    """
    # TODO: Outside the grid boundaries, the interpolating function does
    # nearest neighbor interpolation. This might be a problem if the emcee
    # sampler does not know how to climb a hill back into the grid boundaries.
    # Maybe you want to do a quadratic fit instead, or just set prior boundaries.
    q_grid = lnp_of_ql_grid[:, 0, 0]
    l_grid = lnp_of_ql_grid[0, :, 1]
    lnp_grid = lnp_of_ql_grid[:, :, 2]
    lnp_of_ql = interpolate.RectBivariateSpline(
        q_grid, l_grid, lnp_grid,
        bbox=[None, None, None, None], kx=2, ky=2, s=0)
    return lnp_of_ql.ev


################################################################################
#    Save and load the gridded pseudolikelihoods lnp(q, lambdat)               #
################################################################################

# def save_pseudolikelihood_data(filename, mc_mean_list, lnp_of_ql_grid_list):
#     """Save the data needed to construct the pseudolikelihoods
#     for each of the n BNS systems.
#     """
#     f = h5py.File(filename)
#     nbns = len(mc_mean_list)
#     for i in range(nbns):
#         groupname = 'bns_{}'.format(i)
#         group = f.create_group(groupname)
#         group.attrs['mc_mean'] = mc_mean_list[i]
#         group['lnp_of_ql_grid'] = lnp_of_ql_grid_list[i]
#     f.close()
#
#
# def load_pseudolikelihood_data(filename):
#     """Load the data needed to construct the pseudolikelihoods
#     for each of the n BNS systems.
#     """
#     f = h5py.File(filename)
#     mc_mean_list = []
#     lnp_of_ql_grid_list = []
#     nbns = len(f.keys())
#     for i in range(nbns):
#         # Get data
#         groupname = 'bns_{}'.format(i)
#         mc = f[groupname].attrs['mc_mean']
#         lnp = f[groupname]['lnp_of_ql_grid'][:]
#         # add data to lists
#         mc_mean_list.append(mc)
#         lnp_of_ql_grid_list.append(lnp)
#     f.close()
#     return mc_mean_list, lnp_of_ql_grid_list


def save_pseudolikelihood_data(filename, mc_q_lambdat_list, lnp_of_ql_grid_list):
    """Save the data needed to construct the pseudolikelihoods
    for each of the n BNS systems.
    """
    f = h5py.File(filename)
    nbns = len(mc_q_lambdat_list)
    for i in range(nbns):
        groupname = 'bns_{}'.format(i)
        group = f.create_group(groupname)
        mc_mean = np.mean(mc_q_lambdat_list[i][:, 0])
        group.attrs['mc_mean'] = mc_mean
        group['mc_q_lambdat'] = mc_q_lambdat_list[i]
        group['lnp_of_ql_grid'] = lnp_of_ql_grid_list[i]
    f.close()


def load_pseudolikelihood_data(filename):
    """Load the data needed to construct the pseudolikelihoods
    for each of the n BNS systems.
    """
    f = h5py.File(filename)
    mc_mean_list = []
    mc_q_lambdat_list = []
    lnp_of_ql_grid_list = []
    nbns = len(f.keys())
    for i in range(nbns):
        # Get data
        groupname = 'bns_{}'.format(i)
        mc = f[groupname].attrs['mc_mean']
        mc_q_lambdat = f[groupname]['mc_q_lambdat'][:]
        lnp = f[groupname]['lnp_of_ql_grid'][:]
        # add data to lists
        mc_mean_list.append(mc)
        mc_q_lambdat_list.append(mc_q_lambdat)
        lnp_of_ql_grid_list.append(lnp)
    f.close()
    return mc_mean_list, mc_q_lambdat_list, lnp_of_ql_grid_list
