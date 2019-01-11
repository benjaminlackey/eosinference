import numpy as np
import scipy.interpolate as interpolate


################################################################################
# Class for interpolating a D-dimensional histogram.                           #
################################################################################

def get_bin_index(x, edges):
    """Get the edges index i that contains the point x.
    Point will be between edges[i] and edges[i-1].
    If the point is less than edges[0], then return i=0 (between edges[0] and edges[1]).
    If the point is less than edges[-1], then return nedges-2  (between edges[-2] and edges[-1]).

    Parameters
    ----------
    x : float
    edges : 1d array

    Returns
    -------
    i : int
    """
    # Index of first element greater than x
    iabove = np.argmax(edges>x)
    i = iabove - 1
    # Protect against the left most point
    i = max(i, 0)
    # Protect against the right most point
    nedges = len(edges)
    i = min(i, nedges-2)

    return i


class InterpolateHistogramDD(object):
    """Construct a d-dimensional histogram and interpolate it.
    """
    def __init__(self, hist, edges):
        """
        Parameters
        ----------
        hist : ndim-array histogram
        edges : List of ndim arrays describing the bin edges for each dimension.
        """
        self.hist = hist
        self.edges = edges

    @classmethod
    def from_samples(cls, samples, edges):
        """
        Parameters
        ----------
        samples : 2d array of samples
        edges : List of ndim arrays describing the bin edges for each dimension.
        """
        hist, edges = np.histogramdd(samples, bins=edges)
        return cls(hist, edges)

    def __call__(self, point, extrapolate=0.0):
        """Interpolate at a point.

        Parameters
        ----------
        point : 1d-array of length ndim
        extrapolate : {'edge', float}
            Value to use when extrapolating outside the rectangular boundary of the histogram.
            'edge': Nearest edge index of each coordinate.
            float: Value outside boundary
        """
        ndim = len(point)
        # Test if point is outside edges
        out = np.array([
            (point[i]<self.edges[i][0]) or (point[i]>self.edges[i][-1])
            for i in range(ndim)]).any()
        # For each coordinate of point, get the index of edges just below it
        ixs = tuple([get_bin_index(point[i], self.edges[i]) for i in range(ndim)])
        # Get histogram value f(point)
        f = self.hist[ixs]

        # Return f, or extrapolate if outside region defined by edges
        if out==False:
            return f
        else:
            if extrapolate=='edge':
                return f
            else:
                return extrapolate


################################################################################
# Reweight posterior samples by prior samples                                  #
################################################################################

def reweight_posterior_from_samples(df_prior_old, df_post, edges, columns=['mc', 'q', 'lam_tilde']):
    """Take samples for the old prior and the posterior,
    then reweight the posterior to have a flat prior.
    This is done by multiplying each posterior sample theta by the weight
    weight = 1/prior(theta). The prior function prior(theta) is approximated
    by a d-dimensional histogram over the parameters theta.

    Warning: You must have a LOT of prior samples when making a high-dimensional
    histogram to approximate prior(theta)

    Parameters
    ----------
    df_prior_old : pd.DataFrame containing the old prior samples
    df_post : pd.DataFrame containing the old posterior samples to be reweighted
    edges : List of arrays describing the bin edges for each parameter.
    columns : Names of the columns for each parameter you want to reweight.

    Returns
    -------
    weights : 1d array
        Use these values as the weights when making histograms (or KDEs)
        of the marginalized posterior.
    """
    samples_prior_old = df_prior_old[columns].values
    samples_post = df_post[columns].values
    npost = len(samples_post)

    # Represent the prior function prior(mc, q, lam_tilde) with a histogram.
    prior_func = InterpolateHistogramDD.from_samples(samples_prior_old, edges)

    # Interpolate the prior at the location of the posterior samples.
    prior_at_post = np.array([prior_func(samples_post[i], extrapolate=0.0) for i in range(npost)])

    # Make a histogram of the reweighted posterior
    # Points outside the old prior shouldn't have support in the new prior, so set them to 0.
    weights = np.zeros(npost)
    mask = prior_at_post>0
    weights[mask] = 1.0/prior_at_post[mask]

    return weights


################################################################################
# Reweight posterior samples by prior samples, then marginalize over parameters#
################################################################################

def reweight_posterior_from_samples_lamtilde(df_prior_old, df_post, edges, columns=['mc', 'q', 'lam_tilde']):
    """Take samples for the old prior and the posterior,
    and reweight the posterior to have a flat prior.
    This is done by dividing by the old prior.
    Marginalize over ['mc', 'q'] to get a posterior in 'lam_tilde'.

    Parameters
    ----------
    df_prior_old : pd.DataFrame containing the old prior samples
    df_post : pd.DataFrame containing the old posterior samples to be reweighted
    edges : List of 3 arrays describing the bin edges for each dimension.
    columns : names of the 3 columns

    Returns
    -------
    hist : Reweighted histogram of 'lam_tilde'
    edges_lt : edges of 'lam_tilde' bins
    """
    weights = reweight_posterior_from_samples(
        df_prior_old, df_post, edges, columns=columns)

    # Marginalize over the other parameters [mc, q] with a weighted histogram
    edges_lt = edges[2]
    lamts = df_post[columns[2]].values
    hist, _ = np.histogram(lamts, bins=edges_lt, weights=weights, density=True)

    return hist, edges_lt


def reweight_posterior_from_samples_q_lamtilde(df_prior_old, df_post, edges, columns=['mc', 'q', 'lam_tilde']):
    """Take samples for the old prior and the posterior,
    and reweight the posterior to have a flat prior.
    This is done by dividing by the old prior.
    Marginalize over 'mc' to get a posterior in ['q', 'lam_tilde'].

    Parameters
    ----------
    df_prior_old : pd.DataFrame containing the old prior samples
    df_post : pd.DataFrame containing the old posterior samples to be reweighted
    edges : List of 3 arrays describing the bin edges for each dimension.
    columns : names of the 3 columns

    Returns
    -------
    hist : 2d reweighted histogram of 'q', 'lam_tilde'
    edges_q : edges of 'q' bins
    edges_lt : edges of 'lam_tilde' bins
    """
    weights = reweight_posterior_from_samples(
        df_prior_old, df_post, edges, columns=columns)

    # Marginalize over the other parameters [mc, q] with a weighted histogram
    edges_q = edges[1]
    edges_lt = edges[2]
    qs = df_post[columns[1]].values
    lamts = df_post[columns[2]].values
    hist, _, _ = np.histogram2d(qs, lamts, bins=[edges_q, edges_lt], weights=weights)

    # The histogram2d density keyword only works in numpy 1.15 or later,
    # so do normalize manually
    dq = edges_q[1] - edges_q[0]
    dlt = edges_lt[1] - edges_lt[0]
    hist_norm = dq * dlt * np.sum(hist)
    hist /= hist_norm

    return hist, edges_q, edges_lt


################################################################################
#     Resample distributions from 1d or 2d histograms                          #
################################################################################

def resample_from_histogram(bin_values, bin_edges, ntry):
    """Take a numpy histogram and resample using rejection sampling.

    Parameters
    ----------
    bin_values : shape (nbins, )
        Height of each bin.
    bin_edges : shape (nbins+1, )
        Edges of each bin.
    ntry : int
        Number samples to draw. Only a fraction of these will be accepted.

    Returns
    -------
    x_resamp : shape (naccepted, )
        The resampled points that were accepted.
    """
    # Convert histogram to function with 0th order interpolation.
    # This holds the value from the left side of the histogram until the right side.
    # Points outside the domain are extrapolated to 0 (as it should for a probability density).
    pdf_of_x = interpolate.interp1d(bin_edges[:-1], bin_values, kind='zero', bounds_error=False, fill_value=0.0)

    # Uniformly sample points x in [x_min, x_max] and y in [0, max_height_of_histogram]
    x_min = np.min(bin_edges)
    x_max = np.max(bin_edges)
    max_post =  np.max(bin_values)
    x_prop = np.random.uniform(x_min, x_max, ntry)
    y_prop = np.random.uniform(0.0, max_post, ntry)

    # Reject all samples that are above the histogram
    pdf_at_x_prop = pdf_of_x(x_prop)
    mask = y_prop < pdf_at_x_prop
    x_resamp = x_prop[mask]

    return x_resamp


def resample_from_histogram2d(hist, edges, ntry):
    """Take a numpy histogram2d and resample using rejection sampling.

    Parameters
    ----------
    hist : 2d array
        Height of each bin.
    edges :  List of two 1d arrays describing the edges of each bin [edges_x, edges_y]
        Edges of each bin.
    ntry : int
        Number samples to draw. Only a fraction of these will be accepted.

    Returns
    -------
    xy_samples : 2d array
        The resampled points that were accepted.
    """
    # Interpolate the histogram
    pdf_interp = InterpolateHistogramDD(hist, edges)

    # Uniformly sample points in 3d volume
    # x in [x_min, x_max]
    # y in [y_min, y_max]
    # z in [0, max_height_of_histogram]
    x_min = np.min(edges[0])
    x_max = np.max(edges[0])
    y_min = np.min(edges[1])
    y_max = np.max(edges[1])
    z_max = np.max(hist)
    x_prop = np.random.uniform(x_min, x_max, ntry)
    y_prop = np.random.uniform(y_min, y_max, ntry)
    z_prop = np.random.uniform(0.0, z_max, ntry)

    # Reject all samples that are above the histogram2d
    pdf_at_prop = np.array([
        pdf_interp(np.array([x_prop[i], y_prop[i]]))
        for i in range(ntry)])

    mask = z_prop < pdf_at_prop
    x_resamp = x_prop[mask]
    y_resamp = y_prop[mask]

    xy_samples = np.array([x_resamp, y_resamp]).T
    return xy_samples
