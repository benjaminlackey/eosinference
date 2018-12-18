import numpy as np

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






# def interpolate_histogramdd(point, hist, edges, extrapolate=0.0):
#     """Interpolate a function defined by (hist, edges) = np.histogramdd()
#     at the location point.
#
#     Parameters
#     ----------
#     point : 1d-array of length ndim
#     hist : ndim-array
#     edges : List of ndim arrays describing the bin edges for each dimension.
#     extrapolate : {'edge', float}
#         Value to use when extrapolating outside the rectangular boundary of the histogram.
#         'edge': Nearest edge index of each coordinate.
#         float: Value outside boundary
#     """
#     ndim = len(point)
#     # Test if point is outside edges
#     out = np.array([(point[i]<edges[i][0]) or (point[i]>edges[i][-1]) for i in range(ndim)]).any()
#     # For each coordinate of point, get the index of edges just below it
#     ixs = tuple([get_bin_index(point[i], edges[i]) for i in range(ndim)])
#     # Get histogram value f(point)
#     f = hist[ixs]
#
#     # Return f, or extrapolate if outside region defined by edges
#     if out==False:
#         return f
#     else:
#         if extrapolate=='edge':
#             return f
#         else:
#             return extrapolate
#
#
# class InterpolateHistogramDD(object):
#     """Construct a d-dimensional histogram and interpolate it.
#     """
#     def __init__(self, hist, edges):
#         """
#         Parameters
#         ----------
#         hist : dd array histogram
#         edges : List of ndim arrays describing the bin edges for each dimension.
#         """
#         self.hist = hist
#         self.edges = edges
#
#     @classmethod
#     def from_samples(cls, samples, edges):
#         """
#         Parameters
#         ----------
#         samples : 2d array of samples
#         edges : List of ndim arrays describing the bin edges for each dimension.
#         """
#         hist, edges = np.histogramdd(samples, bins=edges)
#         return cls(hist, edges)
#
#     def __call__(self, point, extrapolate=0.0):
#         """Interpolate at a point.
#
#         Parameters
#         ----------
#         point : 1d-array of coordinates
#         extrapolate : Value to use when extrapolating outside the rectangular boundary of the histogram.
#         """
#         return interpolate_histogramdd(point, self.hist, self.edges, extrapolate=extrapolate)


################################################################################
# Reweight posterior samples by prior samples                                  #
################################################################################


def reweight_posterior_from_samples(df_prior_old, df_post, edges, columns=['mc', 'q', 'lam_tilde']):
    """Take samples for the old prior and the posterior,
    and reweight the posterior to have a flat prior.
    This is done by multiplying each posterior sample (mc, q, lam_tilde) by
    weight = 1/prior(mc, q, lam_tilde).

    Parameters
    ----------
    df_prior_old : pd.DataFrame containing the old prior samples
    df_post : pd.DataFrame containing the old posterior samples to be reweighted
    edges : List of 3 arrays describing the bin edges for each dimension.
    columns : names of the 3 columns

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


# def reweight_posterior_from_samples(df_prior_old, df_post, edges, columns=['mc', 'q', 'lam_tilde']):
#     """Take samples for the old prior and the posterior,
#     and reweight the posterior to have a flat prior.
#     This is done by dividing by the old prior.
#
#     Parameters
#     ----------
#     df_prior_old : pd.DataFrame containing the old prior samples
#     df_post : pd.DataFrame containing the old posterior samples to be reweighted
#     edges : List of 3 arrays describing the bin edges for each dimension.
#     columns : names of the 3 columns
#
#     Returns
#     -------
#     hist_post_reweight : 3d histogram of reweighted posterior
#     """
#     samples_prior_old = df_prior_old[columns].values
#     samples_post = df_post[columns].values
#     npost = len(samples_post)
#
#     # Represent the prior function prior(mc, q, lam_tilde) with a histogram.
#     prior_func = InterpolateHistogramDD(samples_prior_old, edges)
#
#     # Interpolate the prior at the location of the posterior samples.
#     prior_at_post = np.array([prior_func(samples_post[i], extrapolate=0.0) for i in range(npost)])
#
#     # Make a histogram of the reweighted posterior
#     # Points outside the old prior shouldn't have support in the new prior, so set them to 0.
#     weights = np.zeros(npost)
#     mask = prior_at_post>0
#     weights[mask] = 1.0/prior_at_post[mask]
#     hist_post_reweight, _ = np.histogramdd(samples_post, bins=edges, weights=weights)
#
#     return hist_post_reweight


# def reweight_posterior_from_samples_lamtilde(df_prior_old, df_post, edges, columns=['mc', 'q', 'lam_tilde']):
#     """Take samples for the old prior and the posterior,
#     and reweight the posterior to have a flat prior.
#     This is done by dividing by the old prior.
#     Marginalize over ['mc', 'q'] to get a posterior in 'lam_tilde'.
#
#     Parameters
#     ----------
#     df_prior_old : pd.DataFrame containing the old prior samples
#     df_post : pd.DataFrame containing the old posterior samples to be reweighted
#     edges : List of 3 arrays describing the bin edges for each dimension.
#     columns : names of the 3 columns
#
#     Returns
#     -------
#     hist : Reweighted histogram of 'lam_tilde'
#     edges_lt : edges of 'lam_tilde' bins
#     """
#     hist_post_reweight = reweight_posterior_from_samples(
#         df_prior_old, df_post, edges, columns=columns)
#
#     # Integrate over the other parameters [mc, q]
#     sum_axis = (0, 1)
#     hist = np.sum(hist_post_reweight, axis=sum_axis)
#
#     # Convert to proper normalized posterior
#     edges_lt = edges[2]
#     dlt = edges_lt[1] - edges_lt[0]
#     hist_norm = dlt * np.sum(hist)
#     hist /= hist_norm
#
#     return hist, edges_lt


# def reweight_posterior_from_samples_q_lamtilde(df_prior_old, df_post, edges, columns=['mc', 'q', 'lam_tilde']):
#     """Take samples for the old prior and the posterior,
#     and reweight the posterior to have a flat prior.
#     This is done by dividing by the old prior.
#     Marginalize over 'mc' to get a posterior in ['q', 'lam_tilde'].
#
#     Parameters
#     ----------
#     df_prior_old : pd.DataFrame containing the old prior samples
#     df_post : pd.DataFrame containing the old posterior samples to be reweighted
#     edges : List of 3 arrays describing the bin edges for each dimension.
#     columns : names of the 3 columns
#
#     Returns
#     -------
#     hist : 2d reweighted histogram of 'q', 'lam_tilde'
#     edges_q : edges of 'q' bins
#     edges_lt : edges of 'lam_tilde' bins
#     """
#     hist_post_reweight = reweight_posterior_from_samples(
#         df_prior_old, df_post, edges, columns=columns)
#
#     # Integrate over the parameter 'mc'
#     sum_axis = (0, )
#     hist = np.sum(hist_post_reweight, axis=sum_axis)
#
#     # Convert to proper normalized posterior
#     edges_q = edges[1]
#     edges_lt = edges[2]
#     dq = edges_q[1] - edges_q[0]
#     dlt = edges_lt[1] - edges_lt[0]
#     hist_norm = dq * dlt * np.sum(hist)
#     hist /= hist_norm
#
#     return hist, edges_q, edges_lt
