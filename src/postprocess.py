import numpy as np
import matplotlib.pyplot as plt

import utilities as util

################################################################################
#                   Extract information from emcee runs                        #
################################################################################

def downsample_emcee_run(samples, nburnin=0, nthin=1, nsample='all'):
    """Take 3d data from an emcee run, and return a flattened (2d) chain of the
    parameters.

    Parameters
    ----------
    samples : 3d array of samples from emcee
    nburnin : Number of burnin samples to remove from the start of each chain.
    nthin : Stride (number of iterations to skip) when downsampling.
    nsample : {None, int}
        Randomly select this many samples from the flattened chain.
        'all' means keep all of the samples after burnin and thinning.

    Returns
    -------
    samples_flat : downsampled, flattened 2d array of samples
    """
    # Remove burnin, thin the chains, then flatten to single chain
    dim = samples.shape[2]
    samples_flat = samples[:, nburnin::nthin, :].reshape((-1, dim))

    if nsample=='all':
        return samples_flat
    elif (type(nsample)==int) & (nsample>0):
        ndown = len(samples_flat)
        if ndown<nsample:
            print '{} samples were requested. Only {} samples are left. Using {} samples instead.'.format(nsample, ndown, ndown)
            return samples_flat
        else:
            # Get ndown random sample of remaining points
            mask = np.random.choice(np.arange(ndown), size=nsample, replace=False)
            return samples_flat[mask]
    else:
        raise TypeError("nsample must be 'all' or a positive integer.")


def q_eos_samples_from_emcee_samples(samples, nburnin=0, nthin=1, nsample=1000, dim_eos=4):
    """Take 3d data from an emcee run, and return flattened (2d) chains of the
    mass ratios and the eos parameters.
    """
    samples_flat = downsample_emcee_run(samples, nburnin=nburnin, nthin=nthin, nsample=nsample)

    q_samples = samples_flat[:, :-dim_eos]
    eos_samples = samples_flat[:, -dim_eos:]

    return q_samples, eos_samples


def plot_emcee_chains(lnprob, samples, labels=None, truths=None):
    """Plot the chains for each walker and each parameter.
    Also, plot the ln(probability) value for each walker.
    """
    dim = samples.shape[2]

    fig, axes = plt.subplots(
        dim+1, 1,
        figsize=(16, 1.5*(dim+1)), sharex=True, gridspec_kw={'hspace':0.1})

    # Plot posterior chains
    ax = axes[0]
    lnp = lnprob.T
    lnp_max = np.max(lnp)
    ax.plot(lnp, lw=0.5)
    ax.set_ylim(lnp_max-20, lnp_max+1)
    ax.set_ylabel(r'ln(post)')

    # Plot chain for each parameter
    for i in range(dim):
        ax = axes[i+1]
        chains = samples[:, :, i].T
        ax.plot(chains, lw=0.5)
        if labels is not None:
            ax.set_ylabel(labels[i])
        if truths is not None:
            ax.axhline(truths[i], color='r', lw=1)
        ax.minorticks_on()
    axes[-1].set_xlabel(r'Iteration')

    return fig, axes


def compare_2_runs(samples1, samples2, xlabels=None, truths=None, label1=None, label2=None):
    """Make histograms of each parameter. Plot the results of two different runs
    on top of each other.
    """
    dim = samples1.shape[1]
    fig, axes = plt.subplots(dim, 1, figsize=(12, 5*dim))

    for i in range(dim):
        ax = axes[i]

        xs = samples1[:, i]
        ax.hist(xs, bins=20, histtype='step', density=True, label=label1)
        xs = samples2[:, i]
        ax.hist(xs, bins=20, histtype='step', density=True, label=label2)

        ax.legend()
        if xlabels:
            ax.set_xlabel(xlabels[i])
        if truths is not None:
            ax.axvline(truths[i], color='k', lw=2, ls='--')

    return fig, axes


################################################################################
#                 Calculate derived quantities from samples.                   #
# R(M) and Lambda(M) curves, (m, R, Lambda) samples for each NS in each binary #
################################################################################

def ns_properties_from_eos_samples(eos_samples, eos_class_reference, ms):
    """For each EOS sample, calculate R(M) and Lambda(M) curves, maxmass.

    Parameters
    ----------
    eos_samples : 2d array of EOS parameters
    eos_class_reference : Name of the EOS class
    ms : 1d array of masses at which to calculate R(M) and Lambda(M)
    """
    radius_samples = []
    lambda_samples = []
    mmax_samples = []
    eos_valid_samples = []

    for i in range(len(eos_samples)):
        eos_params = eos_samples[i]
        eos = eos_class_reference(eos_params)

        try:
            mmax = eos.max_mass()
            # Set the radius and tidal parameter to 0 if the mass is above mmax
            rs = np.zeros(len(ms))
            ls = np.zeros(len(ms))
            for j in range(len(ms)):
                m = ms[j]
                if m <= mmax:
                    rs[j] = eos.radiusofm(m)
                    ls[j] = eos.lambdaofm(m)
            radius_samples.append(rs)
            lambda_samples.append(ls)
            mmax_samples.append(mmax)
            eos_valid_samples.append(eos_params)
        except RuntimeError:
            print 'LAL had a RuntimeError. Not adding point to samples.'

    return {'mass':ms,
            'radius':np.array(radius_samples),
            'lambda':np.array(lambda_samples),
            'mmax':np.array(mmax_samples),
            'eos':np.array(eos_valid_samples)}


def single_event_ns_properties_from_samples(mc_mean, q_samples, eos_samples, eos_class_reference):
    """Calculate the masses, radii, tidal parameters of the 2 NSs for a given BNS event.
    """
    ns_properties = []
    for i in range(len(q_samples)):
        q = q_samples[i]
        eos_params = eos_samples[i]

        # Get individual masses (with m1>=m2)
        eta = util.eta_of_q(q)
        m1 = util.m1_of_mchirp_eta(mc_mean, eta)
        m2 = util.m2_of_mchirp_eta(mc_mean, eta)

        # Get radii
        try:
            eos = eos_class_reference(eos_params)
            r1 = eos.radiusofm(m1)
            r2 = eos.radiusofm(m2)
            l1 = eos.lambdaofm(m1)
            l2 = eos.lambdaofm(m2)
            lambdat = util.lamtilde_of_eta_lam1_lam2(eta, l1, l2)
            ns_properties.append([m1, m2, r1, r2, l1, l2, q, lambdat])
        except RuntimeError:
            print 'LAL had a RuntimeError. Not adding point to samples.'

    p = np.array(ns_properties)
    return {'m1':p[:, 0], 'm2':p[:, 1],
            'r1':p[:, 2], 'r2':p[:, 3],
            'l1':p[:, 4], 'l2':p[:, 5],
            'q':p[:, 6], 'lambdat':p[:, 7]}


################################################################################
#               Generate contour data for R(M) and Lambda(M)                   #
################################################################################

def mr_contour_data(m1, r1, m2, r2, gridsize=250):
    """
    """
    kde_bound_limits = [m1.min(), 3.0, 0, 20]
    grid_limits = [m1.min()-0.1, m1.max()+0.1, r1.min()-0.1, r1.max()+0.1]
    cd1 = util.estimate_2d_post(m1, r1, kde_bound_limits, grid_limits, gridsize=gridsize)

    kde_bound_limits = [0.5, m2.max(), 0, 20]
    grid_limits = [m2.min()-0.1, m2.max()+0.1, r2.min()-0.1, r2.max()+0.1]
    cd2 = util.estimate_2d_post(m2, r2, kde_bound_limits, grid_limits, gridsize=gridsize)

    return {'cd1':cd1, 'cd2':cd2}


def mlambda_contour_data(m1, l1, m2, l2, gridsize=250):
    """
    """
    kde_bound_limits = [m1.min(), 3.0, 0, 5000]
    #grid_limits = [m1.min()-0.1, m1.max()+0.1, l1.min()-0.1, l1.max()+0.1]
    grid_limits = [m1.min()-0.1, m1.max()+0.1, 0.0, 1.1*l1.max()]
    cd1 = util.estimate_2d_post(m1, l1, kde_bound_limits, grid_limits, gridsize=gridsize)

    kde_bound_limits = [0.5, m2.max(), 0, 5000]
    grid_limits = [m2.min()-0.1, m2.max()+0.1, 0.0, 1.1*l2.max()]
    cd2 = util.estimate_2d_post(m2, l2, kde_bound_limits, grid_limits, gridsize=gridsize)

    return {'cd1':cd1, 'cd2':cd2}


################################################################################
#                         Functions for plotting bounds.                       #
################################################################################

def hpd_interval(x, percentile=90):
    """Get highest posterior density (HPD) interval
    from the minimum interval containing the requested percentile of samples.
    """
    nsamp = len(x)

    # Make sure samples are sorted
    xsort = np.sort(x)

    # Stride containing approximately percentile of samples
    di = int(nsamp*percentile/100.0)

    # Interval and separation for every possible choice of percentile bounds
    intervals = np.array([[xsort[ihigh-di], xsort[ihigh]] for ihigh in range(di, nsamp)])
    dintervals = intervals[:, 1] - intervals[:, 0]

    # Choose the smallest interval. This will be the HPD interval
    intervalmin = np.argmin(dintervals)
    xlow = intervals[intervalmin, 0]
    xhigh = intervals[intervalmin, 1]
    return xlow, xhigh


def bounds_from_curves(ms, curves, percentiles=[50, 90, 100]):
    """Place symmetric upper and lower bounds on a quantity for each mass in ms.

    Parameters
    ----------
    ms : 1d array
        List of masses the curves are evaluated at.
    curves : 2d-arrays
        Each row is a curve of the quantity x(mass).
    percentiles : List
        Percentiles to calculate bounds for.
    """

    bounds = []
    for p in percentiles:
        # Get lower and upper bounds for each symmetric percentile
        half_out = (100.0-p)/2.0
        plow = half_out
        phigh = 100.0-half_out

        d = {'p':p, 'ms':ms}
        xlows = []
        xhighs = []
        for j in range(len(ms)):
            m = ms[j]
            # the value x(m) for each curve at the mass m
            xs = curves[:, j]
            # lower and upper percentiles at the mass m
            xlows.append(np.percentile(xs, plow))
            xhighs.append(np.percentile(xs, phigh))

        d['lows'] = xlows
        d['highs'] = xhighs
        bounds.append(d)
    return bounds
