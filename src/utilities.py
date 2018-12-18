import numpy as np
import pandas as pd

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# LAL
import lal
import lalsimulation

# 1D and 2D bounded KDE
from bounded_2d_kde import Bounded_2d_kde
from bounded_1d_kde import Bounded_1d_kde


################################################################################
#                            binary parameters                                 #
################################################################################

def mchirp_of_m1_m2(m1, m2):
    return (m1*m2)**(3.0/5.0) / (m1+m2)**(1.0/5.0)


def eta_of_m1_m2(m1, m2):
    return (m1*m2) / (m1+m2)**2.0


def eta_of_q(q):
    """Takes either big Q=m_1/m_2 or little q=m_2/m_1 and returns
    symmetric mass ratio eta.
    """
    return q / (1.0 + q)**2


def m1_of_mchirp_eta(mchirp, eta):
    """m1 is always the more massive star (the primary)
    """
    return (1.0/2.0)*mchirp*eta**(-3.0/5.0) * (1.0 + np.sqrt(1.0-4.0*eta))


def m2_of_mchirp_eta(mchirp, eta):
    """m2 is always the less massive star (the secondary)
    """
    return (1.0/2.0)*mchirp*eta**(-3.0/5.0) * (1.0 - np.sqrt(1.0-4.0*eta))


def big_and_small_q_of_eta(eta):
    bigq = (1.0-2.0*eta + np.sqrt(1.0-4.0*eta))/(2.0*eta)
    smallq = (1.0-2.0*eta - np.sqrt(1.0-4.0*eta))/(2.0*eta)
    return bigq, smallq


def lamtilde_of_eta_lam1_lam2(eta, lam1, lam2):
    """$\tilde\Lambda(\eta, \Lambda_1, \Lambda_2)$.
    Lambda_1 is assumed to correspond to the more massive (primary) star m_1.
    Lambda_2 is for the secondary star m_2.
    """
    return (8.0/13.0)*((1.0+7.0*eta-31.0*eta**2)*(lam1+lam2) + np.sqrt(1.0-4.0*eta)*(1.0+9.0*eta-11.0*eta**2)*(lam1-lam2))


def deltalamtilde_of_eta_lam1_lam2(eta, lam1, lam2):
    """This is the definition found in Les Wade's paper.
    Les has factored out the quantity \sqrt(1-4\eta). It is different from Marc Favata's paper.
    $\delta\tilde\Lambda(\eta, \Lambda_1, \Lambda_2)$.
    Lambda_1 is assumed to correspond to the more massive (primary) star m_1.
    Lambda_2 is for the secondary star m_2.
    """
    return (1.0/2.0)*(
                      np.sqrt(1.0-4.0*eta)*(1.0 - 13272.0*eta/1319.0 + 8944.0*eta**2/1319.0)*(lam1+lam2)
                      + (1.0 - 15910.0*eta/1319.0 + 32850.0*eta**2/1319.0 + 3380.0*eta**3/1319.0)*(lam1-lam2)
                      )


def lam1_lam2_of_pe_params(eta, lamt, dlamt):
    """lam1 is for the the primary mass m_1.
    lam2 is for the the secondary mass m_2.
    m_1 >= m2.
    """

    a = (8.0/13.0)*(1.0+7.0*eta-31.0*eta**2)
    b = (8.0/13.0)*np.sqrt(1.0-4.0*eta)*(1.0+9.0*eta-11.0*eta**2)
    c = (1.0/2.0)*np.sqrt(1.0-4.0*eta)*(1.0 - 13272.0*eta/1319.0 + 8944.0*eta**2/1319.0)
    d = (1.0/2.0)*(1.0 - 15910.0*eta/1319.0 + 32850.0*eta**2/1319.0 + 3380.0*eta**3/1319.0)

    den = (a+b)*(c-d) - (a-b)*(c+d)

    lam1 = ( (c-d)*lamt - (a-b)*dlamt )/den
    lam2 = (-(c+d)*lamt + (a+b)*dlamt )/den

    return lam1, lam2


################################################################################
# Sample the prior for an MCMC run.                                            #
################################################################################

def mass_tidal_prior(m1bounds, m2bounds, mcbounds, qbounds, lambda1bounds, lambda2bounds, ntry=500000):
    """Generate a prior that is flat in m1, m2, lambda1, lambda2.
    Additional bounds are determined by mcbounds, qbounds.
    ntry : Number of samples to generate before making additional cuts.
    """
    m1 = np.random.uniform(m1bounds[0], m1bounds[1], ntry)
    m2 = np.random.uniform(m2bounds[0], m2bounds[1], ntry)
    lam1 = np.random.uniform(lambda1bounds[0], lambda1bounds[1], ntry)
    lam2 = np.random.uniform(lambda2bounds[0], lambda2bounds[1], ntry)

    # Make DataFrame with all quantities
    mc = mchirp_of_m1_m2(m1, m2)
    q = m2/m1
    eta = eta_of_q(q)
    lam_tilde = lamtilde_of_eta_lam1_lam2(eta, lam1, lam2)
    dlam_tilde = deltalamtilde_of_eta_lam1_lam2(eta, lam1, lam2)
    prior_df = pd.DataFrame(
        {'m1':m1, 'm2':m2, 'mc':mc, 'q':q, 'eta':eta,
        'lambda1':lam1, 'lambda2':lam2, 'lam_tilde':lam_tilde, 'dlam_tilde':dlam_tilde})

    # Make additional cuts
    # chirp mass bounds
    mcmin, mcmax = mcbounds
    prior_df = prior_df[(prior_df['mc']>=mcmin) & (prior_df['mc']<=mcmax)]
    # mass ratio bounds
    # If qmax<=1, then m1>=m2
    qmin, qmax = qbounds
    prior_df = prior_df[(prior_df['q']>=qmin) & (prior_df['q']<=qmax)]

    return prior_df



################################################################################
#         Methods for plotting 1 and 2 dimensional distributions               #
################################################################################


################################ 1D ##############################


def hist_rescale_to_one(axes, x, bins, **kwargs):
    """Plot a histogram with the maximum value rescaled to 1.

    Parameters
    ----------
    axes : matplotlib axes object
    x : numpy array of samples
    bins : Edges of bins
    **kwargs : used in axes.hist(**kwargs)
    """
    npoints = len(x)

    # Make a histogram and get the maximum bin height
    hist, xedges = np.histogram(x, bins=bins)
    hist_max = hist.max()

    # The value of each point (normally 1) is reweighted.
    weights = np.ones(npoints)/hist_max

    # Draw the histogram
    axes.hist(x, bins=bins, weights=weights, **kwargs)


def bounded_kde_rescale_to_one(
    axes, x, x_prior=None, xlow=None, xhigh=None, grid_low=None, grid_high=None,
    gridsize=1000, alpha=0.2, label=None, transpose=False, bw_method=None, **kwargs):
    """Plot a bounded kde with maximum value rescaled to 1.

    Parameters
    ----------
    axes : matplotlib axes object
    x : numpy array of samples
    **kwargs : used in axes.plot()
    """
    # The 1d KDE object annoyingly requires data to be passed as 2d not 1d.

    # Initialize bounded KDE object
    post_kde = Bounded_1d_kde(np.atleast_2d(x).T, xlow=xlow, xhigh=xhigh,
                                bw_method=bw_method)
    if x_prior is not None:
        prior_kde = Bounded_1d_kde(np.atleast_2d(x_prior).T, xlow=xlow, xhigh=xhigh,
                                    bw_method=bw_method)

    # Evaluate KDE on a grid
    x_grid = np.linspace(grid_low, grid_high, gridsize)
    post_grid = post_kde(np.atleast_2d(x_grid).T)
    if x_prior is not None:
        prior_grid = prior_kde(np.atleast_2d(x_grid).T)

    # Rescale so that the prior is flat
    if x_prior is not None:
        post_grid = post_grid / prior_grid

    # Rescale max of KDE to 1
    post_scaled = post_grid / np.max(post_grid)

    if transpose==False:
        axes.plot(x_grid, post_scaled, label=label, **kwargs)
        axes.fill_between(x_grid, post_scaled, alpha=alpha, **kwargs)
    else:
        axes.plot(post_scaled, x_grid, label=label, **kwargs)
        axes.fill_betweenx(x_grid, post_scaled, alpha=alpha, **kwargs)


################################ 2D ##############################


def estimate_2d_post(x, y, kde_bound_limits, grid_limits, gridsize=500, bw_method=None):
    """Estimate a 2d posterior from samples (x, y)
    using a Bounded_2d_kde with boundaries kde_bound_limits=[xlow, xhigh, ylow, yhigh].
    Calculate this posterior on a grid with boundaries grid_limits=[xlow, xhigh, ylow, yhigh].

    Parameters
    ----------

    Returns
    -------
    xx : x values on 2d grid (x is first index, y is second index)
    yy : y values on 2d grid
    z_grid : Normalized posterior on 2d grid
    z_points : Normalized posterior at original samples
    """
    # Initialize KDE
    points = np.array([x, y]).T
    xlow, xhigh, ylow, yhigh = kde_bound_limits
    #print points
    #print xlow, xhigh, ylow, yhigh
    post_kde = Bounded_2d_kde(points, xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh,
                              bw_method=bw_method)

    # Make a list of [x, y] coordinates for the grid
    x_grid = np.linspace(grid_limits[0], grid_limits[1], gridsize)
    y_grid = np.linspace(grid_limits[2], grid_limits[3], gridsize)
    points_grid = np.array([[[x, y] for y in y_grid] for x in x_grid])

    # x values on 2d grid (x is first index, y is second index)
    xx = points_grid[:, :, 0]

    # y values on 2d grid
    yy = points_grid[:, :, 1]

    # Evaluate KDE on grid
    points_grid = points_grid.reshape(gridsize*gridsize, 2)
    z_grid = post_kde(points_grid)
    z_grid = z_grid.reshape(gridsize, gridsize)

    # Evaluate KDE at original samples x, y
    z_points = post_kde(points)

    return {'xx':xx, 'yy':yy, 'z_grid':z_grid, 'z_points':z_points}


def contour_value_from_posterior(post, percentile):
    """Get contour value from the list of *marginalized* 2d-posteriors
    evaluated at the samples (e.g. evaluated with a 2d-KDE from the 2d samples).
    The contour will contain percentile/100 fraction of the samples.
    """
    if (percentile < 0.0) | (percentile > 100.0):
        raise Exception, 'percentile must be between 0 and 100.'

    nsamp = len(post)
    # Samples reverse sorted
    post_rev = np.sort(post)[::-1]
    percentile_i = int(np.floor(nsamp*percentile/100.0))
    # Don't go beyond last sample in case you use percentile=100
    # or something close to 100.
    percentile_i = min(percentile_i, nsamp-1)

    return post_rev[percentile_i]


def plot_posterior_with_contours(
    ax, contour_data, cmap=None, shade_level=95, levels=[50, 90], white_contour_back=True,
    colors=['k', 'r'], linestyles=['-', '--'], linewidths=[1, 1],
    interpolation='bilinear'):
    """
    Parameters
    ----------
    """
    xx = contour_data['xx']
    yy = contour_data['yy']
    zz = contour_data['z_grid']
    post = contour_data['z_points']

    # Plot shading. Mask shading beyond shade_level.
    if cmap!=None:
        shade_bound = contour_value_from_posterior(post, shade_level)
        zmasked = np.ma.masked_where(zz<shade_bound, zz)
        extent = [xx[0, 0], xx[-1, 0], yy[0, 0], yy[0, -1]]
        ax.imshow(
            zmasked.T,
            interpolation=interpolation, origin='lower',
            extent=extent, aspect='auto', cmap=cmap)

    # Get contour values from percentiles
    zvalues = np.array([contour_value_from_posterior(post, p) for p in levels])

    # Plot white background behind contours (to mask shading)
    if white_contour_back==True:
        ax.contour(
            xx, yy, zz, zvalues[::-1],
            colors='w', linewidths=2*np.array(linewidths[::-1]))

    # Plot contours
    ax.contour(
        xx, yy, zz, zvalues[::-1],
        colors=colors[::-1], linestyles=linestyles[::-1], linewidths=linewidths[::-1])


################################ 2d triangle plot ##############################

def triangle_plot_2d_axes(
    xbounds, ybounds, figsize=(8, 8),
    width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.0, hspace=0.0,
    grid=False):
    """Initialize the axes for a 2d triangle plot.
    """
    high1d = 1.05

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=width_ratios, height_ratios=height_ratios,
        wspace=wspace, hspace=hspace)

    ax1 = plt.subplot(gs[0])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])

    ax1.minorticks_on()
    ax3.minorticks_on()
    ax4.minorticks_on()

    if grid:
        ax1.grid(which='major', ls='-')
        ax1.grid(which='minor', ls=':')
        ax3.grid(which='major', ls='-')
        ax3.grid(which='minor', ls=':')
        ax4.grid(which='major', ls='-')
        ax4.grid(which='minor', ls=':')

    # Get rid of tick labels
    ax1.xaxis.set_ticklabels([])
    ax4.yaxis.set_ticklabels([])

    # Use consistent x-axis and y-axis bounds in all 3 plots
    ax1.set_ylim(0, high1d)
    ax1.set_xlim(xbounds[0], xbounds[1])
    ax3.set_xlim(xbounds[0], xbounds[1])
    ax3.set_ylim(ybounds[0], ybounds[1])
    ax4.set_xlim(0, high1d)
    ax4.set_ylim(ybounds[0], ybounds[1])

    return fig, ax1, ax3, ax4
